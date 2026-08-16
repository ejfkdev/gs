// bge.go - 纯 Go BGE-small-zh-v1.5 encoder (inference only)
// 架构: 4 层 BERT, 512 hidden, 8 heads, FFN=2048
//
// 权重读取见 safetensors.go: 直接读 HuggingFace 原始 model.safetensors

package gs

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"sync"
)

// ---------- BGE 模型 ----------

type BGEConfig struct {
	VocabSize int
	Hidden    int
	Layers    int
	Heads     int
	FFN       int
	MaxPos    int
	TypeVocab int
	HeadDim   int // = Hidden / Heads
}

type BGEWeights struct {
	Cfg BGEConfig

	WordEmb    []float32 // V × H
	PosEmb     []float32 // P × H
	TypeEmb    []float32 // TV × H
	EmbLnGamma []float32 // H
	EmbLnBeta  []float32 // H

	Layers  []BGELayerWeights
	PoolerW []float32 // H × H
	PoolerB []float32 // H
}

type BGELayerWeights struct {
	QW, QB  []float32
	KW, KB  []float32
	VW, VB  []float32
	OW, OB  []float32
	AttnLnG []float32
	AttnLnB []float32
	FFN1W   []float32
	FFN1B   []float32
	FFN2W   []float32
	FFN2B   []float32
	FFNLnG  []float32
	FFNLnB  []float32
}

// loadBGE: 读取 safetensors 格式的 BGE 权重 (唯一支持的权重格式)
func loadBGE(path string) (*BGEWeights, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return parseSafetensors(data)
}

// ---------- 数学操作 ----------

// layerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
func layerNorm(x, gamma, beta, y []float32, n int, eps float32) {
	var sum float32
	for i := 0; i < n; i++ {
		sum += x[i]
	}
	mean := sum / float32(n)
	var sqsum float32
	for i := 0; i < n; i++ {
		d := x[i] - mean
		sqsum += d * d
	}
	var_ := sqsum / float32(n)
	invStd := 1.0 / float32(math.Sqrt(float64(var_+eps)))
	for i := 0; i < n; i++ {
		y[i] = (x[i]-mean)*invStd*gamma[i] + beta[i]
	}
}

func gelu(x float32) float32 {
	return 0.5 * x * (1.0 + float32(math.Erf(float64(x)*0.7071067811865475)))
}

// ---------- Encoder ----------

// Encode: tokens (CLS + token + SEP) → pooler output
// 走批量通道 (B=1), 委托给 EncodeBatch 复用 blas32 sgemm 优化
func (w *BGEWeights) Encode(tokenIDs []int) []float32 {
	return w.EncodeBatch([][]int{tokenIDs})[0]
}

// ---------- WordPiece Tokenizer (BGE-BERT) ----------

type WordPieceTokenizer struct {
	vocab      map[string]int
	idToToken  []string
	clsID      int
	sepID      int
	padID      int
	unkID      int
	maxLen     int
	basicRuned []rune // CJK 字符表 (basic)
}

func loadTokenizer(vocabPath string) (*WordPieceTokenizer, error) {
	f, err := os.Open(vocabPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	t := &WordPieceTokenizer{
		vocab:  make(map[string]int, 22000),
		maxLen: 512,
	}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			continue
		}
		t.vocab[line] = len(t.vocab)
	}
	t.idToToken = make([]string, len(t.vocab))
	for k, v := range t.vocab {
		t.idToToken[v] = k
	}
	t.clsID = t.vocab["[CLS]"]
	t.sepID = t.vocab["[SEP]"]
	t.padID = t.vocab["[PAD]"]
	t.unkID = t.vocab["[UNK]"]
	if _, ok := t.vocab["[CLS]"]; !ok {
		return nil, fmt.Errorf("[CLS] not in vocab")
	}
	t.basicRuned = []rune{0x4E00, 0x9FFF}
	return t, nil
}

// Tokenize: BERT-style
// 1. 切分 (CJK char-per-char, 其他按空格/标点)
// 2. WordPiece: 贪心 longest-match, 否则 [UNK]
func (t *WordPieceTokenizer) Tokenize(text string) []string {
	var tokens []string
	runes := []rune(text)
	i := 0
	for i < len(runes) {
		r := runes[i]
		// 跳过空白
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			i++
			continue
		}
		// CJK 标点 (0x3000-0x303F): 跳过, 单独成 token
		if r >= 0x3000 && r <= 0x303F {
			tokens = append(tokens, string(r))
			i++
			continue
		}
		// CJK: 每个字符一个 token
		if r >= 0x4E00 && r <= 0x9FFF {
			tokens = append(tokens, string(r))
			i++
			continue
		}
		// 其他: 收集连续的非空白 unicode
		start := i
		for i < len(runes) {
			r := runes[i]
			if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
				break
			}
			if r >= 0x4E00 && r <= 0x9FFF {
				break
			}
			if r >= 0x3000 && r <= 0x303F { // CJK 标点
				break
			}
			i++
		}
		if i > start {
			word := string(runes[start:i])
			tokens = append(tokens, word)
		}
	}

	// WordPiece: 对非 CJK 的 word 做最长匹配
	var out []string
	for _, tok := range tokens {
		if len(tok) > 0 {
			r := []rune(tok)
			if r[0] >= 0x4E00 && r[0] <= 0x9FFF {
				out = append(out, tok) // CJK 直接
				continue
			}
			out = append(out, t.wordPiece(tok)...)
		}
	}
	return out
}

func (t *WordPieceTokenizer) wordPiece(word string) []string {
	var result []string
	start := 0
	runes := []rune(word)
	for start < len(runes) {
		end := len(runes)
		var cur string
		for end > start {
			sub := string(runes[start:end])
			if start > 0 {
				sub = "##" + sub
			}
			if _, ok := t.vocab[sub]; ok {
				cur = sub
				break
			}
			end--
		}
		if cur == "" {
			return []string{"[UNK]"}
		}
		result = append(result, cur)
		start = end
	}
	return result
}

// TokenizeToIDs: text → [CLS] + tokens + [SEP]
func (t *WordPieceTokenizer) TokenizeToIDs(text string) []int {
	toks := t.Tokenize(text)
	ids := make([]int, 0, len(toks)+2)
	ids = append(ids, t.clsID)
	for _, tok := range toks {
		if id, ok := t.vocab[tok]; ok {
			ids = append(ids, id)
		} else {
			ids = append(ids, t.unkID)
		}
		if len(ids) >= t.maxLen-1 {
			break
		}
	}
	ids = append(ids, t.sepID)
	return ids
}

// ---------- BGE Engine 整合 ----------

// bgeCacheMax: query → embedding 缓存上限 (超过后整体清空, 防止长驻进程
// 因 query 无限增长耗尽内存)
const bgeCacheMax = 4096

// DefaultMaxEmbedRunes: BGE 编码时每个字段截断的默认最大 rune 数。
// bge-small-zh-v1.5 支持 512 token; 对中文是 1 汉字 = 1 token, 512 rune
// 正好吃满模型上下文。对英文可以调得更高 (Schema.MaxEmbedRunes)。
const DefaultMaxEmbedRunes = 512

// embedBatchSize: EmbedBatch 的内部分批大小 (实测 blas32 sgemm 在 B=8 时收益最好)
const embedBatchSize = 8

// effectiveMaxRunes: schema 值 <=0 时回退到默认
func effectiveMaxRunes(v int) int {
	if v <= 0 {
		return DefaultMaxEmbedRunes
	}
	return v
}

// BGEEngine: BGE encoder + cache
type BGEEngine struct {
	Weights   *BGEWeights
	Tokenizer *WordPieceTokenizer
	maxRunes  int // 编码时每个字段截断的最大 rune 数
	cacheMu   sync.Mutex
	cache     map[string][]float32
}

// NewBGEEngine: 从文件加载 weights + vocab
func NewBGEEngine(weightsPath, vocabPath string) (*BGEEngine, error) {
	w, err := loadBGE(weightsPath)
	if err != nil {
		return nil, err
	}
	t, err := loadTokenizer(vocabPath)
	if err != nil {
		return nil, err
	}
	return &BGEEngine{
		Weights:   w,
		Tokenizer: t,
		maxRunes:  DefaultMaxEmbedRunes,
		cache:     make(map[string][]float32),
	}, nil
}

// Embed: text → 归一化的 512 维向量 (带缓存, 供 query 路径使用)
func (b *BGEEngine) Embed(text string) []float32 {
	b.cacheMu.Lock()
	if v, ok := b.cache[text]; ok {
		b.cacheMu.Unlock()
		return v
	}
	b.cacheMu.Unlock()

	vec := b.encode(text)

	b.cacheMu.Lock()
	if len(b.cache) >= bgeCacheMax {
		b.cache = make(map[string][]float32, bgeCacheMax)
	}
	b.cache[text] = vec
	b.cacheMu.Unlock()
	return vec
}

// EmbedFresh: text → 向量, 不走缓存也不写缓存 (build 路径用, 避免
// 为海量 item 文本积累无用的缓存条目)
func (b *BGEEngine) EmbedFresh(text string) []float32 {
	return b.encode(text)
}

// encode: 截断 + tokenize + forward pass
func (b *BGEEngine) encode(text string) []float32 {
	text = truncateRunes(text, b.maxRunes)
	ids := b.Tokenizer.TokenizeToIDs(text)
	return b.Weights.Encode(ids)
}

// EmbedBatch: 批量 text → B 个 512 维向量
// 自动利用 cache (跳过已编码的 text), 剩余未缓存的按 batched 编码
func (b *BGEEngine) EmbedBatch(texts []string) [][]float32 {
	if len(texts) == 0 {
		return nil
	}
	out := make([][]float32, len(texts))
	pendingIdx := make([]int, 0, len(texts))
	pendingIDs := make([][]int, 0, len(texts))

	for i, text := range texts {
		t := truncateRunes(text, b.maxRunes)
		b.cacheMu.Lock()
		if v, ok := b.cache[t]; ok {
			b.cacheMu.Unlock()
			out[i] = v
			continue
		}
		b.cacheMu.Unlock()
		ids := b.Tokenizer.TokenizeToIDs(t)
		pendingIdx = append(pendingIdx, i)
		pendingIDs = append(pendingIDs, ids)
	}

	if len(pendingIDs) == 0 {
		return out
	}

	// 把 pending 按 T 排序, 减少 padding 浪费
	pairs := make([]embedBatchPair, len(pendingIDs))
	for i, ids := range pendingIDs {
		pairs[i] = embedBatchPair{origIdx: pendingIdx[i], ids: ids, tLen: len(ids)}
	}
	sortPairsByLen(pairs)

	for start := 0; start < len(pairs); start += embedBatchSize {
		end := start + embedBatchSize
		if end > len(pairs) {
			end = len(pairs)
		}
		batch := pairs[start:end]
		batchIDs := make([][]int, len(batch))
		for i, p := range batch {
			batchIDs[i] = p.ids
		}
		vecs := b.Weights.EncodeBatch(batchIDs)
		for i, p := range batch {
			out[p.origIdx] = vecs[i]
		}
	}

	return out
}

// embedBatchPair: EmbedBatch 内部用的排序单元
type embedBatchPair struct {
	origIdx int
	ids     []int
	tLen    int
}

// sortPairsByLen: 按 tLen 升序排序 (减少 padding 浪费)
func sortPairsByLen(pairs []embedBatchPair) {
	// insertion sort
	for i := 1; i < len(pairs); i++ {
		j := i
		for j > 0 && pairs[j-1].tLen > pairs[j].tLen {
			pairs[j-1], pairs[j] = pairs[j], pairs[j-1]
			j--
		}
	}
}

// EmbedDim: 输出维度
func (b *BGEEngine) EmbedDim() int {
	if b == nil || b.Weights == nil {
		return 0
	}
	return b.Weights.Cfg.Hidden
}

// CacheStats: 返回 cache 大小
func (b *BGEEngine) CacheStats() int {
	b.cacheMu.Lock()
	defer b.cacheMu.Unlock()
	return len(b.cache)
}

// ResetCache: 清空 cache
func (b *BGEEngine) ResetCache() {
	b.cacheMu.Lock()
	b.cache = make(map[string][]float32)
	b.cacheMu.Unlock()
}
