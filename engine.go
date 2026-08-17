// engine.go - 多字段混合搜索引擎
// 核心算法: per-field BM25 + per-field BGE embedding + phrase + exact + prefix + strict
// + Stage 2 rerank (token overlap + per-field embedding cos)

package gs

import (
	"context"
	"hash/fnv"
	"math"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// 默认权重
const (
	DefaultEmbDim    = 512
	DefaultTopK      = 10
	DefaultCacheSize = 256
	DefaultRerankTop = 100
	DefaultRerankW   = 0.20
	DefaultWBM25     = 0.40
	DefaultWEmb      = 0.45
	DefaultWPhrase   = 0.10
	DefaultWExact    = 0.20
	DefaultWPrefix   = 0.10
	DefaultWStrict   = 0.30
	BM25K1           = 1.5
	BM25B            = 0.75
	ItemsMagic       = 0x48494B53 // "SKIH" little-endian
	EmbMagic         = 0x424D454D // "EMBM"
	FormatVer        = 1

	// 索引目录内 BGE 模型/词表的固定文件名。Load 按此自动读取, Build 按此写入。
	// 权重直接用 HuggingFace 原始 safetensors 格式, 不再有自定义转换格式。
	ModelFile = "model.safetensors"
	VocabFile = "vocab.txt"
)

// PostingList: 一个 term 在一个 field 下的所有 posting
type PostingList struct {
	DocIDs []uint32
	TFs    []uint16
}

// hashTerm: FNV-1a 64 位 hash
func hashTerm(s string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(s))
	return h.Sum64()
}

// internalItem: 引擎内部的 item 表示 (从 items.bin 加载)
type internalItem struct {
	ID        string
	Path      string
	Source    string
	Tags      []string
	FieldOffs []uint32 // 长度 = field_count
	PathOff   uint32
	TagsOff   uint32
	SourceOff uint32
}

// stage1Hit: stage 1 排序的中间结果
type stage1Hit struct {
	idx    int
	score  float32
	strict []string
}

// Engine: 多字段混合搜索引擎
type Engine struct {
	schema     *Schema
	dataDir    string
	n          int
	fieldCount int

	items      []*internalItem
	fieldsBuf  [][]byte // per-field string buffer (null-terminated)
	pathsBuf   []byte
	tagsBuf    []byte
	sourcesBuf []byte

	// Inverted index
	invIdx map[uint64]map[int]*PostingList // term_hash -> field_idx -> postings
	// Per-field BM25 stats
	fieldDocLens  [][]uint16 // [field_idx][doc_idx]
	fieldAvgD     []float64  // [field_idx]
	fieldTotalLen []int      // [field_idx]
	bm25N         int

	// Embeddings
	embField [][]float32 // [field_idx][doc_idx * dim]
	embDim   int

	// "Name" field index (for exact/prefix matching)
	nameFieldIdx int      // -1 if not present
	nameLower    []string // 预计算的 name 字段小写 (Load 时构建)

	// BGE encoder + LRU cache
	bge   *BGEEngine
	cache *LRUCache
}

// NewEngine: 构造 (但不加载, 等待 Load 填充数据)
func NewEngine(schema *Schema) *Engine {
	return &Engine{
		schema:       schema,
		fieldCount:   len(schema.Fields),
		invIdx:       make(map[uint64]map[int]*PostingList, 50000),
		nameFieldIdx: schema.primaryFieldIndex(),
		cache:        NewLRU(),
	}
}

// Fields: 返回 schema 字段
func (e *Engine) Fields() []Field {
	if e.schema == nil {
		return nil
	}
	return e.schema.Fields
}

// Schema: 返回 schema
func (e *Engine) Schema() *Schema {
	return e.schema
}

// DataDir: 返回 dataDir
func (e *Engine) DataDir() string {
	return e.dataDir
}

// Close: 释放资源
func (e *Engine) Close() error {
	if e.bge != nil {
		e.bge.ResetCache()
		e.bge = nil
	}
	return nil
}

// ---------- 字符串缓冲读取 ----------

// readStr: 从 buffer 读 null-terminated 字符串
func readStr(buf []byte, off uint32) string {
	if int(off) >= len(buf) {
		return ""
	}
	end := off
	for end < uint32(len(buf)) && buf[end] != 0 {
		end++
	}
	return string(buf[off:end])
}

// fieldText: 读 item 的某个 field 文本
func (e *Engine) fieldText(itemIdx, fieldIdx int) string {
	if itemIdx < 0 || itemIdx >= e.n || fieldIdx < 0 || fieldIdx >= e.fieldCount {
		return ""
	}
	return readStr(e.fieldsBuf[fieldIdx], e.items[itemIdx].FieldOffs[fieldIdx])
}

// pathText: 读 item 的 path
func (e *Engine) pathText(itemIdx int) string {
	if itemIdx < 0 || itemIdx >= e.n {
		return ""
	}
	return readStr(e.pathsBuf, e.items[itemIdx].PathOff)
}

// sourceText: 读 item 的 source
func (e *Engine) sourceText(itemIdx int) string {
	if itemIdx < 0 || itemIdx >= e.n {
		return ""
	}
	return readStr(e.sourcesBuf, e.items[itemIdx].SourceOff)
}

// nameAt: item 的 name 字段小写 (idx < 0 时返回空)
func (e *Engine) nameAt(itemIdx int) string {
	if e.nameFieldIdx < 0 {
		return ""
	}
	if e.nameLower != nil {
		return e.nameLower[itemIdx]
	}
	return lowerASCII(e.fieldText(itemIdx, e.nameFieldIdx))
}

// ---------- BM25: per-field ----------

// buildInvertedIndex: 从已加载的 items 构建 per-field inverted index (并行)
func (e *Engine) buildInvertedIndex() {
	e.fieldDocLens = make([][]uint16, e.fieldCount)
	e.fieldAvgD = make([]float64, e.fieldCount)
	e.fieldTotalLen = make([]int, e.fieldCount)
	for fi := 0; fi < e.fieldCount; fi++ {
		e.fieldDocLens[fi] = make([]uint16, e.n)
	}
	e.bm25N = e.n

	workers := runtime.NumCPU()
	if workers > e.n {
		workers = e.n
	}
	if workers < 1 {
		workers = 1
	}
	// 每个 worker 算自己的 local inv idx + local docLens + local totalLen
	type localIndex struct {
		invIdx   map[uint64]map[int]*PostingList
		docLens  [][]uint16
		totalLen []int
	}
	locals := make([]localIndex, workers)
	for w := 0; w < workers; w++ {
		locals[w].invIdx = make(map[uint64]map[int]*PostingList, 50000)
		locals[w].docLens = make([][]uint16, e.fieldCount)
		for fi := 0; fi < e.fieldCount; fi++ {
			locals[w].docLens[fi] = make([]uint16, e.n)
		}
		locals[w].totalLen = make([]int, e.fieldCount)
	}

	var wg sync.WaitGroup
	workCh := make(chan int, workers*2)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(workerIdx int) {
			defer wg.Done()
			local := &locals[workerIdx]
			for i := range workCh {
				for fi := 0; fi < e.fieldCount; fi++ {
					text := e.fieldText(i, fi)
					if text == "" {
						continue
					}
					tfCount := make(map[string]int, 32)
					for _, t := range tokenize(text) {
						tfCount[t]++
					}
					local.docLens[fi][i] = uint16(len(tfCount))
					local.totalLen[fi] += len(tfCount)
					for tok, tf := range tfCount {
						h := hashTerm(tok)
						m, ok := local.invIdx[h]
						if !ok {
							m = make(map[int]*PostingList, 1)
							local.invIdx[h] = m
						}
						lst, ok := m[fi]
						if !ok {
							lst = &PostingList{}
							m[fi] = lst
						}
						lst.DocIDs = append(lst.DocIDs, uint32(i))
						lst.TFs = append(lst.TFs, uint16(tf))
					}
				}
			}
		}(w)
	}
	for i := 0; i < e.n; i++ {
		workCh <- i
	}
	close(workCh)
	wg.Wait()

	// 合并 worker 的 docLens + totalLen (同一 item 只会被一个 worker 处理)
	for fi := 0; fi < e.fieldCount; fi++ {
		total := 0
		for w := 0; w < workers; w++ {
			for i, v := range locals[w].docLens[fi] {
				if v > e.fieldDocLens[fi][i] {
					e.fieldDocLens[fi][i] = v
				}
			}
			total += locals[w].totalLen[fi]
		}
		e.fieldTotalLen[fi] = total
		if e.bm25N > 0 {
			e.fieldAvgD[fi] = float64(total) / float64(e.bm25N)
		}
	}

	// 合并 inverted index (term -> field -> postings; docID 不重叠, 直接 append)
	for w := 0; w < workers; w++ {
		for h, m := range locals[w].invIdx {
			globalM, ok := e.invIdx[h]
			if !ok {
				globalM = make(map[int]*PostingList, len(m))
				e.invIdx[h] = globalM
			}
			for fi, localLst := range m {
				globalLst, ok := globalM[fi]
				if !ok {
					globalLst = &PostingList{}
					globalM[fi] = globalLst
				}
				globalLst.DocIDs = append(globalLst.DocIDs, localLst.DocIDs...)
				globalLst.TFs = append(globalLst.TFs, localLst.TFs...)
			}
		}
	}
}

// section: 一个 item 在所有 field 上的 tokenize + posting 构建 (worker 内联路径)

// bm25SearchField: 在单个 field 上做 BM25
func (e *Engine) bm25SearchField(qToks []string, fieldIdx int) []float32 {
	scores := make([]float32, e.n)
	if fieldIdx < 0 || fieldIdx >= e.fieldCount {
		return scores
	}
	avgD := e.fieldAvgD[fieldIdx]
	if avgD == 0 {
		return scores
	}
	dlArr := e.fieldDocLens[fieldIdx]
	for _, qt := range qToks {
		perField, ok := e.invIdx[hashTerm(qt)]
		if !ok {
			continue
		}
		lst, ok := perField[fieldIdx]
		if !ok {
			continue
		}
		df := float64(len(lst.DocIDs))
		idf := math.Log((float64(e.bm25N)-df+0.5)/(df+0.5) + 1.0)
		for i, docID := range lst.DocIDs {
			tf := float64(lst.TFs[i])
			dl := float64(dlArr[docID])
			if dl == 0 {
				dl = avgD
			}
			tfNorm := tf * (BM25K1 + 1) / (tf + BM25K1*(1-BM25B+BM25B*dl/avgD))
			scores[docID] += float32(idf * tfNorm)
		}
	}
	return scores
}

// bm25SearchFields: 在多个 fields 上做 BM25, 按 FieldWeight 加权合并
func (e *Engine) bm25SearchFields(qToks []string, fields []int) []float32 {
	scores := make([]float32, e.n)
	for _, fi := range fields {
		if fi < 0 || fi >= e.fieldCount {
			continue
		}
		w := e.schema.Fields[fi].FieldWeight
		if w == 0 {
			w = 1.0
		}
		s := e.bm25SearchField(qToks, fi)
		for i := range scores {
			scores[i] += s[i] * w
		}
	}
	return scores
}

// ---------- BGE Embedding ----------

// embedQuery: BGE 编码 query
func (e *Engine) embedQuery(text string) []float32 {
	if e.bge != nil {
		return e.bge.Embed(text)
	}
	// fallback: char-trigram hash embedding (BGE 模型不可用时)
	return hashEmbed(text, e.embDim)
}

// hashEmbed: 占位的 hash embedding (用于 BGE 不可用时)
func hashEmbed(text string, dim int) []float32 {
	if dim == 0 {
		dim = DefaultEmbDim
	}
	v := make([]float32, dim)
	text = strings.ToLower(text)
	for i := 0; i+2 < len(text); i++ {
		v[hashNgram(text[i:i+3])] += 1.0
	}
	for _, w := range strings.Fields(text) {
		if len(w) >= 2 {
			v[hashNgram(w)] += 2.0
		}
	}
	toks := strings.Fields(text)
	for i := 0; i+1 < len(toks); i++ {
		v[hashNgram(toks[i]+" "+toks[i+1])] += 3.0
	}
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	if s > 0 {
		norm := float32(1.0 / math.Sqrt(s))
		for i := range v {
			v[i] *= norm
		}
	}
	return v
}

func hashNgram(s string) int {
	h := fnv.New32a()
	h.Write([]byte(s))
	return int(h.Sum32() % uint32(DefaultEmbDim))
}

// embSearchMultiField: 用 per-field embedding 算 cos, 按 field weight 加权
func (e *Engine) embSearchMultiField(query string, fields []int) []float32 {
	scores := make([]float32, e.n)
	qEmb := e.embedQuery(query)
	if len(qEmb) == 0 || len(e.embField) == 0 {
		return scores
	}
	dim := e.embDim
	var totalW float32
	for _, fi := range fields {
		if fi < 0 || fi >= e.fieldCount || !e.schema.Fields[fi].Embeddable {
			continue
		}
		w := e.schema.Fields[fi].FieldWeight
		if w == 0 {
			w = 1.0
		}
		totalW += w
	}
	if totalW == 0 {
		return scores
	}
	for _, fi := range fields {
		if fi < 0 || fi >= e.fieldCount || !e.schema.Fields[fi].Embeddable || fi >= len(e.embField) {
			continue
		}
		embs := e.embField[fi]
		if len(embs) < e.n*dim {
			continue
		}
		w := e.schema.Fields[fi].FieldWeight
		if w == 0 {
			w = 1.0
		}
		for i := 0; i < e.n; i++ {
			off := i * dim
			var dot float64
			for j := 0; j < dim; j++ {
				dot += float64(qEmb[j]) * float64(embs[off+j])
			}
			scores[i] += float32(dot) * w
		}
	}
	for i := range scores {
		scores[i] /= totalW
	}
	return scores
}

// embSearchField: 单字段 embedding cos
func (e *Engine) embSearchField(query string, fieldIdx int) []float32 {
	scores := make([]float32, e.n)
	if fieldIdx < 0 || fieldIdx >= len(e.embField) {
		return scores
	}
	qEmb := e.embedQuery(query)
	if len(qEmb) == 0 || len(e.embField[fieldIdx]) < e.n*e.embDim {
		return scores
	}
	dim := e.embDim
	embs := e.embField[fieldIdx]
	for i := 0; i < e.n; i++ {
		off := i * dim
		var dot float64
		for j := 0; j < dim; j++ {
			dot += float64(qEmb[j]) * float64(embs[off+j])
		}
		scores[i] = float32(dot)
	}
	return scores
}

// ---------- Phrase Boost ----------

// containsFold: s 是否包含 sub (ASCII 大小写不敏感; sub 应已小写)
func containsFold(s, sub string) bool {
	if sub == "" || len(sub) > len(s) {
		return false
	}
	i := 0
	for {
		j := indexByteFold(s, sub[0], i)
		if j < 0 || j+len(sub) > len(s) {
			return false // j 之后的首字节出现处也放不下, 提前退出
		}
		ok := true
		for k := 1; k < len(sub); k++ {
			if foldByte(s[j+k]) != sub[k] {
				ok = false
				break
			}
		}
		if ok {
			return true
		}
		i = j + 1
	}
}

func indexByteFold(s string, c byte, start int) int {
	for i := start; i < len(s); i++ {
		if foldByte(s[i]) == c {
			return i
		}
	}
	return -1
}

func foldByte(c byte) byte {
	if c >= 'A' && c <= 'Z' {
		return c + 32
	}
	return c
}

// phraseBoost: 相邻 query token 短语出现在 item 字段中时的加分。
// 只在已经在 BM25 中命中的文档上计算 — 短语命中必然伴随 token 命中,
// 所以不会损失召回, 但把 O(n*fields) 的全文扫描降到了候选集内。
func (e *Engine) phraseBoost(qToks []string, fields []int, restrict []float32) []float32 {
	boosts := make([]float32, e.n)
	if len(qToks) < 2 || len(qToks) > 20 {
		return boosts
	}
	if len(fields) == 0 {
		for fi, f := range e.schema.Fields {
			if f.Searchable {
				fields = append(fields, fi)
			}
		}
	}

	// 收集候选短语 (CJK bigram 直接是文本子串; 拉丁短语用空格连接)
	seen := make(map[string]bool, len(qToks)*2)
	var phrases []string
	add := func(p string) {
		if p != "" && !seen[p] {
			seen[p] = true
			phrases = append(phrases, p)
		}
	}
	for _, t := range qToks {
		if rs := []rune(t); len(rs) == 2 && isCJK(rs[0]) && isCJK(rs[1]) {
			add(t)
		}
	}
	for i := 1; i < len(qToks); i++ {
		a, b := qToks[i-1], qToks[i]
		if a != "" && b != "" {
			add(a + " " + b)
			if i+1 < len(qToks) && qToks[i+1] != "" {
				add(a + " " + b + " " + qToks[i+1])
			}
		}
	}

	for _, phrase := range phrases {
		three := len(strings.Fields(phrase)) >= 3
		for _, fi := range fields {
			for i := 0; i < e.n; i++ {
				if restrict[i] <= 0 {
					continue
				}
				text := e.fieldText(i, fi)
				if text == "" || !containsFold(text, phrase) {
					continue
				}
				if three {
					boosts[i] += 0.5
				} else {
					boosts[i] += 0.4
				}
			}
		}
	}
	return boosts
}

// ---------- Exact / Prefix ----------

// nameMatches: 检查每个 item 的 "name" 字段是否含 query substring 或 token
func (e *Engine) nameMatches(qToks []string, qLower string) []float32 {
	matches := make([]float32, e.n)
	if e.nameFieldIdx < 0 {
		return matches
	}
	for i := 0; i < e.n; i++ {
		nm := e.nameAt(i)
		if strings.Contains(nm, qLower) {
			matches[i] = 0.5
		}
		for _, qt := range qToks {
			if len(qt) >= 2 && strings.Contains(nm, qt) {
				matches[i] += 0.3
			}
		}
	}
	return matches
}

// exactNameMatch: 返回 name 字段与 query 精确相等的 item index (-1 if none)
func (e *Engine) exactNameMatch(query string) int {
	if e.nameFieldIdx < 0 {
		return -1
	}
	for i := 0; i < e.n; i++ {
		if e.fieldText(i, e.nameFieldIdx) == query {
			return i
		}
	}
	return -1
}

// prefixMatches: 返回所有 name 以 query 开头的 item index
func (e *Engine) prefixMatches(qLower string) []int {
	var out []int
	if e.nameFieldIdx < 0 {
		return out
	}
	for i := 0; i < e.n; i++ {
		if strings.HasPrefix(e.nameAt(i), qLower) {
			out = append(out, i)
		}
	}
	return out
}

// ---------- Strict ----------

// strictFieldsForItem: 收集每个 item 的 strict 加权字段 (懒构建; 字符串
// 直接引用 item buffer, 不复制)
func (e *Engine) strictFieldsForItem() [][]StrictField {
	out := make([][]StrictField, e.n)
	nameIdx := e.nameFieldIdx
	for i := 0; i < e.n; i++ {
		fields := make([]StrictField, 0, 4)
		for fi, f := range e.schema.Fields {
			if !f.Strict {
				continue
			}
			fields = append(fields, StrictField{
				Name:   f.Name,
				Text:   e.fieldText(i, fi),
				Weight: 0.15,
			})
		}
		// path 不是 schema 字段, 始终参与 strict (中权重)
		fields = append(fields, StrictField{
			Name:   "path",
			Text:   e.pathText(i),
			Weight: 0.25,
		})
		// name 字段 (强): 仅在 schema 没有把 name 标成 Strict 时才额外补一次,
		// 避免 name 被 f.Strict 循环和这里双重计权。
		if nameIdx >= 0 && !e.schema.Fields[nameIdx].Strict {
			fields = append(fields, StrictField{
				Name:   "name",
				Text:   e.fieldText(i, nameIdx),
				Weight: 0.3,
			})
		}
		out[i] = fields
	}
	return out
}

// ---------- Snippet ----------

// extractSnippet: 从 Snippet=true 字段提 snippet
func (e *Engine) extractSnippet(itemIdx int, query string, fields []int) string {
	snippetFis := fields
	if len(snippetFis) == 0 {
		for fi, f := range e.schema.Fields {
			if f.Snippet {
				snippetFis = append(snippetFis, fi)
			}
		}
	}
	if len(snippetFis) == 0 {
		return ""
	}
	var parts []string
	for _, fi := range snippetFis {
		text := e.fieldText(itemIdx, fi)
		if text == "" {
			continue
		}
		s := ExtractSnippets(text, query, 2, 40, 60)
		if s == "" {
			continue
		}
		if name := e.schema.Fields[fi].Name; name != "" {
			parts = append(parts, "["+name+"] "+s)
		} else {
			parts = append(parts, s)
		}
	}
	return strings.Join(parts, " ... ")
}

// ---------- Normalize ----------

func normalizeScores(scores []float32) []float32 {
	if len(scores) == 0 {
		return scores
	}
	min, max := scores[0], scores[0]
	for _, s := range scores {
		if s < min {
			min = s
		}
		if s > max {
			max = s
		}
	}
	rng := max - min
	if rng == 0 {
		return scores
	}
	out := make([]float32, len(scores))
	for i, s := range scores {
		out[i] = (s - min) / rng
	}
	return out
}

// ---------- 搜索主流程 ----------

// resolveSearchFields: 解析用户指定的 fields (所有子集: searchable/embeddable/snippet)
func (e *Engine) resolveSearchFields(opts SearchOptions) (searchable, embeddable, snippet []int) {
	appendAll := func() {
		for fi, f := range e.schema.Fields {
			if f.Searchable {
				searchable = append(searchable, fi)
			}
			if f.Embeddable {
				embeddable = append(embeddable, fi)
			}
			if f.Snippet {
				snippet = append(snippet, fi)
			}
		}
	}
	if len(opts.Fields) == 0 {
		appendAll()
		return
	}
	nameMap := make(map[string]int, len(e.schema.Fields))
	for fi, f := range e.schema.Fields {
		nameMap[f.Name] = fi
	}
	for _, name := range opts.Fields {
		fi, ok := nameMap[name]
		if !ok {
			continue
		}
		f := e.schema.Fields[fi]
		if f.Searchable {
			searchable = append(searchable, fi)
		}
		if f.Embeddable {
			embeddable = append(embeddable, fi)
		}
		if f.Snippet {
			snippet = append(snippet, fi)
		}
	}
	return
}

// searchCacheKey: 缓存 key 必须覆盖所有影响结果的选项
func searchCacheKey(opts SearchOptions) string {
	var sb strings.Builder
	sb.WriteString(strings.ToLower(opts.Query))
	sb.WriteByte('|')
	sb.WriteString(strconv.Itoa(opts.TopK))
	sb.WriteByte(',')
	sb.WriteString(strconv.Itoa(opts.RerankTop))
	sb.WriteString("|f:")
	sb.WriteString(strings.Join(opts.Fields, ","))
	sb.WriteString("|strict:")
	sb.WriteString(strconv.FormatBool(opts.Strict))
	// 所有可调权重都进 key, 避免改权重后命中旧缓存
	w := func(v float32) string { return strconv.FormatFloat(float64(v), 'f', 2, 32) }
	sb.WriteString("|w:")
	sb.WriteString(w(opts.WBM25))
	sb.WriteByte(',')
	sb.WriteString(w(opts.WEmb))
	sb.WriteByte(',')
	sb.WriteString(w(opts.WPhrase))
	sb.WriteByte(',')
	sb.WriteString(w(opts.WExact))
	sb.WriteByte(',')
	sb.WriteString(w(opts.WPrefix))
	sb.WriteByte(',')
	sb.WriteString(w(opts.WStrict))
	sb.WriteByte(',')
	sb.WriteString(w(opts.RerankW))
	return sb.String()
}

// Search: 搜索主入口
func (e *Engine) Search(ctx context.Context, opts SearchOptions) ([]Hit, error) {
	if e.n == 0 || opts.Query == "" {
		return nil, nil
	}
	if opts.TopK <= 0 {
		opts.TopK = DefaultTopK
	}
	if opts.RerankTop <= 0 {
		opts.RerankTop = DefaultRerankTop
	}
	if opts.RerankW <= 0 {
		opts.RerankW = DefaultRerankW
	}
	if opts.WBM25 == 0 {
		opts.WBM25 = DefaultWBM25
	}
	if opts.WEmb == 0 {
		opts.WEmb = DefaultWEmb
	}
	if opts.WPhrase == 0 {
		opts.WPhrase = DefaultWPhrase
	}
	if opts.WExact == 0 {
		opts.WExact = DefaultWExact
	}
	if opts.WPrefix == 0 {
		opts.WPrefix = DefaultWPrefix
	}
	if opts.WStrict == 0 {
		opts.WStrict = DefaultWStrict
	}

	// 缓存命中 (返回拷贝, 调用方修改不影响缓存)
	qKey := searchCacheKey(opts)
	if cached, ok := e.cache.Get(qKey); ok {
		return cached, nil
	}

	qToks := tokenizeDedup(opts.Query)
	if len(qToks) == 0 {
		return nil, nil
	}

	searchable, embeddable, snippetFis := e.resolveSearchFields(opts)

	// 1. BM25 (per-field, weighted)
	bm25Scores := e.bm25SearchFields(qToks, searchable)
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// 2. BGE per-field
	embScores := e.embSearchMultiField(opts.Query, embeddable)
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// 3. Phrase (只在 BM25 命中的文档上算)
	phraseScores := e.phraseBoost(qToks, searchable, bm25Scores)

	// 4. Exact / Prefix (基于 name field)
	qLower := strings.ToLower(opts.Query)
	nameScores := e.nameMatches(qToks, qLower)
	exactIdx := e.exactNameMatch(opts.Query)
	prefixSet := make(map[int]bool)
	for _, p := range e.prefixMatches(qLower) {
		prefixSet[p] = true
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// 5. Strict
	strictBoosts, strictTypes := computeStrictBoost(opts.Query, e.strictFieldsForItem(), e.n)
	hasStrict := false
	for _, st := range strictTypes {
		if len(st) > 0 {
			hasStrict = true
			break
		}
	}
	strictActive := opts.Strict || hasStrict

	// 归一化
	bm25N := normalizeScores(bm25Scores)
	embN := normalizeScores(embScores)
	phraseN := normalizeScores(phraseScores)
	nameN := normalizeScores(nameScores)
	strictN := normalizeScores(strictBoosts)

	// Stage 1: hybrid score
	stage1 := make([]stage1Hit, e.n)
	for i := 0; i < e.n; i++ {
		score := opts.WBM25*bm25N[i] + opts.WEmb*embN[i] + opts.WPhrase*phraseN[i] + 0.25*nameN[i]
		if strictActive && strictN[i] > 0 {
			score += opts.WStrict * strictN[i]
		}
		if exactIdx == i {
			score += opts.WExact
		} else if prefixSet[i] {
			score += opts.WPrefix
		}
		var st []string
		if i < len(strictTypes) {
			st = strictTypes[i]
		}
		stage1[i] = stage1Hit{idx: i, score: score, strict: st}
	}
	sort.Slice(stage1, func(i, j int) bool { return stage1[i].score > stage1[j].score })
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Stage 2: rerank top-RerankTop
	rerankN := opts.RerankTop
	if rerankN > len(stage1) {
		rerankN = len(stage1)
	}
	candidates := stage1[:rerankN]
	rerankScores := e.rerank(opts.Query, qToks, candidates, embeddable)
	rerankNorm := normalizeScores(rerankScores)
	for i := range candidates {
		candidates[i].score = (1-opts.RerankW)*candidates[i].score + opts.RerankW*rerankNorm[i]
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].score > candidates[j].score })

	// 截 topK
	k := opts.TopK
	if k > len(candidates) {
		k = len(candidates)
	}
	final := candidates[:k]

	// 构建 Hit
	hits := make([]Hit, 0, k)
	for _, c := range final {
		hit := e.buildHit(c.idx, c.score, c.strict, opts.Query, snippetFis, exactIdx == c.idx, prefixSet[c.idx], bm25Scores[c.idx], embScores[c.idx])
		hits = append(hits, hit)
	}

	e.cache.Set(qKey, hits)
	// 返回拷贝: 缓存里保留内部副本, 调用方修改不影响后续命中
	return cloneHits(hits), nil
}

// rerank: 对候选集重排序 (token overlap + per-field embedding cos)
func (e *Engine) rerank(query string, qToks []string, candidates []stage1Hit, embeddable []int) []float32 {
	scores := make([]float32, len(candidates))
	if len(candidates) == 0 {
		return scores
	}
	qTokSet := make(map[string]bool, len(qToks))
	for _, t := range qToks {
		qTokSet[t] = true
	}

	// 找 desc field (第一个 LongText, 用于 descTokHit)
	descFieldIdx := -1
	for fi, f := range e.schema.Fields {
		if f.Type == FieldLongText {
			descFieldIdx = fi
			break
		}
	}

	// 计算 per-field embedding cos (候选 rerank 信号)
	fieldEmbScores := make(map[int][]float32, len(embeddable))
	for _, fi := range embeddable {
		if fi < 0 || fi >= len(e.embField) {
			continue
		}
		fieldEmbScores[fi] = e.embSearchField(query, fi)
	}

	for ci, c := range candidates {
		var sig float32
		// name 字段 token overlap
		nm := e.nameAt(c.idx)
		overlap := 0
		for t := range qTokSet {
			if strings.Contains(nm, t) {
				overlap++
			}
		}
		// 相邻 token 命中
		bigramMatch := 0
		for j := 0; j+1 < len(qToks); j++ {
			if nm != "" && strings.Contains(nm, qToks[j]+qToks[j+1]) {
				bigramMatch++
			}
		}
		// desc 包含 query token 数
		descTokHit := 0
		var descLower string
		if descFieldIdx >= 0 {
			descLower = strings.ToLower(e.fieldText(c.idx, descFieldIdx))
		}
		for _, qt := range qToks {
			if len(qt) >= 2 && strings.Contains(descLower, qt) {
				descTokHit++
			}
		}
		if len(qToks) > 0 {
			sig = float32(overlap)/float32(len(qToks))*0.4 +
				float32(bigramMatch)*0.3 +
				float32(descTokHit)/float32(len(qToks))*0.3
		}
		// strict token 命中加权
		if len(c.strict) > 0 {
			sig += 0.5
		}
		// per-field embedding cos
		var embSig float32
		var embN int
		for _, fi := range embeddable {
			if s, ok := fieldEmbScores[fi]; ok && c.idx < len(s) {
				embSig += (s[c.idx] + 1) / 2
				embN++
			}
		}
		if embN > 0 {
			sig += (embSig / float32(embN)) * 0.3
		}
		scores[ci] = sig
	}
	return scores
}

// buildHit: 从内部 idx 构造 Hit (snippet 始终生成)
func (e *Engine) buildHit(idx int, score float32, strict []string, query string, snippetFis []int, exact, prefix bool, bm25Score, embScore float32) Hit {
	h := Hit{
		ID:     e.items[idx].ID,
		Idx:    idx,
		Score:  score,
		Path:   e.pathText(idx),
		Source: e.sourceText(idx),
		Tags:   append([]string(nil), e.items[idx].Tags...), // 拷贝, 避免调用方改动内部状态
		FTS5:   bm25Score,
		Emb:    embScore,
		Exact:  exact,
		Prefix: prefix,
		Strict: strict,
	}
	h.Fields = make(map[string]string, len(e.schema.Fields))
	for fi, f := range e.schema.Fields {
		if f.Display {
			h.Fields[f.Name] = e.fieldText(idx, fi)
		}
	}
	h.Snippet = e.extractSnippet(idx, query, snippetFis)
	return h
}

// ---------- LRU ----------

// LRUCache: 简单的 LRU 缓存 (Get 返回深拷贝, 调用方修改不影响缓存)
type LRUCache struct {
	mu    sync.Mutex
	items map[string][]Hit
	order []string
	size  int
}

func NewLRU() *LRUCache {
	return &LRUCache{items: make(map[string][]Hit, DefaultCacheSize), size: DefaultCacheSize}
}

func (c *LRUCache) Get(k string) ([]Hit, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	v, ok := c.items[k]
	if !ok {
		return nil, false
	}
	return cloneHits(v), true
}

func (c *LRUCache) Set(k string, v []Hit) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.items[k]; ok {
		c.items[k] = v
		return
	}
	if len(c.order) >= c.size {
		old := c.order[0]
		c.order = c.order[1:]
		delete(c.items, old)
	}
	c.order = append(c.order, k)
	c.items[k] = v
}

// cloneHits: 拷贝 Hits (Tags / Strict 切片和 Fields map 是新分配, 字符串共享不可变)
func cloneHits(hits []Hit) []Hit {
	out := make([]Hit, len(hits))
	for i, h := range hits {
		h.Tags = append([]string(nil), h.Tags...)
		h.Strict = append([]string(nil), h.Strict...)
		if h.Fields != nil {
			m := make(map[string]string, len(h.Fields))
			for k, v := range h.Fields {
				m[k] = v
			}
			h.Fields = m
		}
		out[i] = h
	}
	return out
}

// ---------- Fast Search (无 BGE, 纯 BM25) ----------

// FastSearchResult: 快速搜索结果
type FastSearchResult struct {
	Idx     int
	Path    string
	Source  string
	Name    string
	Desc    string
	Score   float32
	Snippet string
}

// FastSearch: 纯 BM25 搜索, 跳过 BGE (适合把一整个文件当作 query)
func (e *Engine) FastSearch(docText string, k int) []FastSearchResult {
	if k <= 0 {
		k = DefaultTopK
	}
	// 截断超长输入, 和旧实现对齐 (防止一整个大文件拖垮 BM25)
	if len(docText) > 2000 {
		docText = string([]rune(docText)[:2000])
	}
	qToks := tokenizeDedup(docText)
	if len(qToks) == 0 {
		return nil
	}
	var sfs []int
	for fi, f := range e.schema.Fields {
		if f.Searchable {
			sfs = append(sfs, fi)
		}
	}
	scores := e.bm25SearchFields(qToks, sfs)
	order := make([]int, e.n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return scores[order[a]] > scores[order[b]] })

	descIdx := -1
	for fi, f := range e.schema.Fields {
		if f.Type == FieldLongText {
			descIdx = fi
			break
		}
	}
	out := make([]FastSearchResult, 0, k)
	for _, idx := range order {
		if scores[idx] <= 0 || len(out) >= k {
			break
		}
		var name, desc string
		if e.nameFieldIdx >= 0 {
			name = e.fieldText(idx, e.nameFieldIdx)
		}
		if descIdx >= 0 {
			desc = e.fieldText(idx, descIdx)
		}
		out = append(out, FastSearchResult{
			Idx:     idx,
			Path:    e.pathText(idx),
			Source:  e.sourceText(idx),
			Name:    name,
			Desc:    desc,
			Score:   scores[idx],
			Snippet: ExtractSnippets(desc, docText, 2, 40, 60),
		})
	}
	return out
}

// GC 触发 (用于大型引擎定期回收)
func (e *Engine) MaybeGC() {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	if ms.HeapAlloc > 1024*1024*1024 {
		runtime.GC()
	}
}
