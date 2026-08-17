// build.go - 索引构建器
// Builder 收集 items, 然后 Build 写出 items.bin + emb_<field>.bin
// + model.safetensors/vocab.txt (BGE 模型) + schema.json
//
// 流程:
//   1. Add 阶段: 收集 items
//   2. Build 阶段:
//      a. 分配每个 Field 的 string buffer (dedup)
//      b. 算每个 Embeddable Field 的 BGE embedding (并行, 无缓存)
//      c. 写 items.bin / emb_<field>.bin
//      d. 把 BGE 模型/词表复制为固定名 model.safetensors + vocab.txt
//      e. 写 schema.json

package gs

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
)

// internalBuildItem: build 阶段收集的 item (Fields 按下标与 schema 对齐)
type internalBuildItem struct {
	ID     string
	Path   string
	Source string
	Tags   []string
	Fields []string // field_idx -> text
}

// Builder: 索引构建器
type Builder struct {
	schema  *Schema
	dataDir string

	// BGE 配置 (Build 时需要 BGE encoder)
	bgeWeightsPath string
	bgeVocabPath   string
	bge            *BGEEngine

	// 并行度 (默认 runtime.NumCPU())
	workers int

	// 持久化 embedding 缓存目录 (可选; 非空时按字段内容哈希复用向量)
	embCacheDir string

	// 收集的 items
	items []*internalBuildItem

	// 进度回调
	OnProgress func(stage string, cur, total int)
}

// BuilderOption: Builder 配置选项
type BuilderOption func(*Builder)

// WithBGEPaths: 设置 BGE weights + vocab 路径 (Build 时需要)
func WithBGEPaths(weightsPath, vocabPath string) BuilderOption {
	return func(b *Builder) {
		b.bgeWeightsPath = weightsPath
		b.bgeVocabPath = vocabPath
	}
}

// WithProgress: 进度回调
func WithProgress(cb func(stage string, cur, total int)) BuilderOption {
	return func(b *Builder) {
		b.OnProgress = cb
	}
}

// WithWorkers: 设置 BGE 编码并行度 (默认 runtime.NumCPU())
func WithWorkers(n int) BuilderOption {
	return func(b *Builder) {
		if n < 1 {
			n = 1
		}
		b.workers = n
	}
}

// WithEmbedCache: 设置持久化 embedding 缓存目录。非空时按「字段内容哈希」复用
// 向量, 新增/删除/改动文档时只重算变化的部分 (缓存落在 cacheDir, 应比索引目录
// 更持久, 例如 watch 场景放到 output 旁的 .embcache)。
func WithEmbedCache(cacheDir string) BuilderOption {
	return func(b *Builder) {
		b.embCacheDir = cacheDir
	}
}

// NewBuilder: 创建一个 builder (会确保 dataDir 存在)
func NewBuilder(schema *Schema, dataDir string, opts ...BuilderOption) (*Builder, error) {
	if err := schema.Validate(); err != nil {
		return nil, fmt.Errorf("validate schema: %w", err)
	}
	b := &Builder{
		schema:  schema,
		dataDir: dataDir,
		workers: runtime.NumCPU(), // 默认全核
	}
	for _, o := range opts {
		o(b)
	}
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return nil, fmt.Errorf("mkdir dataDir: %w", err)
	}
	return b, nil
}

// workerCount: 实际用的 worker 数
func (b *Builder) workerCount() int {
	if b.workers < 1 {
		return runtime.NumCPU()
	}
	return b.workers
}

// Add: 添加一个 item
func (b *Builder) Add(item Item) error {
	if item.ID == "" {
		item.ID = item.Path // fallback
	}
	if item.Source == "" {
		item.Source = "item"
	}
	ib := &internalBuildItem{
		ID:     item.ID,
		Path:   item.Path,
		Source: item.Source,
		Tags:   item.Tags,
		Fields: make([]string, len(b.schema.Fields)),
	}
	for fi, f := range b.schema.Fields {
		ib.Fields[fi] = item.Fields[f.Name]
	}
	// Item.Tags 合并到每一个 FieldTags 字段 (类型驱动, 与字段名无关)。
	// 注意: 不要把同一份 tags 同时写进 Fields[...] 和 Item.Tags, 否则会重复。
	if item.Tags != nil {
		tagsStr := joinTags(item.Tags)
		if tagsStr != "" {
			for fi, f := range b.schema.Fields {
				if f.Type != FieldTags {
					continue
				}
				if existing := ib.Fields[fi]; existing != "" {
					ib.Fields[fi] = existing + " " + tagsStr
				} else {
					ib.Fields[fi] = tagsStr
				}
			}
		}
	}
	b.items = append(b.items, ib)
	return nil
}

// joinTags: tags → "tag1 tag2 tag3"
func joinTags(tags []string) string {
	if len(tags) == 0 {
		return ""
	}
	out := ""
	for i, t := range tags {
		if i > 0 {
			out += " "
		}
		out += t
	}
	return out
}

// AddCount: 当前已加 item 数
func (b *Builder) AddCount() int {
	return len(b.items)
}

// canReuseEmbedding: 增量 build 支持 — 若 emb_<field>.bin 已存在且大小与
// header (N + dim) 对得上, 说明该字段的 embedding 无需重算。只读 8 字节
// header 判断, 不读整个文件 (Load 会从磁盘重新加载)。
func canReuseEmbedding(path string, n, dim int) bool {
	st, err := os.Stat(path)
	if err != nil {
		return false
	}
	if st.Size() != int64(8)+int64(n)*int64(dim)*4 {
		return false
	}
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	var hdr [8]byte
	if _, err := io.ReadFull(f, hdr[:]); err != nil {
		return false
	}
	return int(binary.LittleEndian.Uint32(hdr[0:])) == n &&
		int(binary.LittleEndian.Uint32(hdr[4:])) == dim
}

// Close: 关闭 (释放 BGE 资源)
func (b *Builder) Close() error {
	if b.bge != nil {
		b.bge.ResetCache()
		b.bge = nil
	}
	return nil
}

// Build: 完成构建, 写所有 .bin 文件
func (b *Builder) Build() error {
	if len(b.items) == 0 {
		return fmt.Errorf("no items added")
	}

	n := len(b.items)
	fieldCount := len(b.schema.Fields)

	// 0. 加载 BGE (如果有 embeddable field)
	needEmb := false
	for _, f := range b.schema.Fields {
		if f.Embeddable {
			needEmb = true
			break
		}
	}
	if needEmb {
		if b.bgeWeightsPath == "" {
			// 默认从 dataDir 找 (增量 rebuild 场景)
			b.bgeWeightsPath = firstExisting(filepath.Join(b.dataDir, ModelFile))
		}
		if b.bgeVocabPath == "" {
			b.bgeVocabPath = firstExisting(filepath.Join(b.dataDir, VocabFile))
		}
		if b.bgeWeightsPath != "" && b.bgeVocabPath != "" {
			bge, err := NewBGEEngine(b.bgeWeightsPath, b.bgeVocabPath)
			if err != nil {
				return fmt.Errorf("init BGE for build: %w", err)
			}
			bge.maxRunes = effectiveMaxRunes(b.schema.MaxEmbedRunes)
			b.bge = bge
		}
		if b.bge == nil {
			// 没有 BGE 模型: 优雅降级为纯 BM25 索引
			needEmb = false
		}
	}

	embDim := DefaultEmbDim
	if b.bge != nil {
		embDim = b.bge.EmbedDim()
	}

	// 1. per-field string interning (去重)
	fieldStrs := make([]map[string]uint32, fieldCount)
	fieldBuf := make([][]byte, fieldCount)
	for i := 0; i < fieldCount; i++ {
		fieldStrs[i] = make(map[string]uint32, 1024)
		fieldBuf[i] = make([]byte, 0, 1024)
	}
	pathsStrs := make(map[string]uint32, 1024)
	pathsBuf := make([]byte, 0, 1024)
	tagsStrs := make(map[string]uint32, 1024)
	tagsBuf := make([]byte, 0, 1024)
	sourcesStrs := make(map[string]uint32, 1024)
	sourcesBuf := make([]byte, 0, 1024)

	itemFieldOffs := make([][]uint32, n)
	itemPathOffs := make([]uint32, n)
	itemTagsOffs := make([]uint32, n)
	itemSourceOffs := make([]uint32, n)

	// 写入字符串到 buffer (null-terminated), 返回 offset
	intern := func(m map[string]uint32, buf *[]byte, s string) uint32 {
		if s == "" {
			return 0
		}
		if off, ok := m[s]; ok {
			return off
		}
		off := uint32(len(*buf))
		m[s] = off
		*buf = append(*buf, s...)
		*buf = append(*buf, 0)
		return off
	}

	b.progress("intern", 0, n)
	for i, it := range b.items {
		itemFieldOffs[i] = make([]uint32, fieldCount)
		for fi := 0; fi < fieldCount; fi++ {
			itemFieldOffs[i][fi] = intern(fieldStrs[fi], &fieldBuf[fi], it.Fields[fi])
		}
		itemPathOffs[i] = intern(pathsStrs, &pathsBuf, it.Path)
		itemTagsOffs[i] = intern(tagsStrs, &tagsBuf, joinTags(it.Tags))
		itemSourceOffs[i] = intern(sourcesStrs, &sourcesBuf, it.Source)
		if i%1000 == 0 {
			b.progress("intern", i, n)
		}
	}
	b.progress("intern", n, n)

	// 2. 写 items.bin
	itemsPath := filepath.Join(b.dataDir, "items.bin")
	if err := writeItemsBin(b, itemsPath, itemFieldOffs, itemPathOffs, itemTagsOffs, itemSourceOffs, fieldBuf, pathsBuf, tagsBuf, sourcesBuf); err != nil {
		return fmt.Errorf("write items.bin: %w", err)
	}
	b.progress("items.bin", 1, 1)

	// 3. BGE embedding (并行单条 + 可选内容哈希缓存增量)
	if needEmb && b.bge != nil {
		for fi, f := range b.schema.Fields {
			if !f.Embeddable {
				continue
			}
			embPath := filepath.Join(b.dataDir, fmt.Sprintf("emb_%s.bin", f.Name))
			// 无缓存时: 整文件大小/header 对上才复用 (N 不变); 有缓存时交给内容哈希复用
			if b.embCacheDir == "" && canReuseEmbedding(embPath, n, embDim) {
				b.progress("embed_"+f.Name, n, n)
				continue
			}

			emb := make([]float32, n*embDim)
			if err := b.embedField(fi, emb, embPath, embDim); err != nil {
				return err
			}
		}
	}

	// 4. BGE 模型/词表落盘 (Load 按固定名自动读取): 权重直接复制 safetensors 原文
	if needEmb && b.bge != nil {
		if err := copyFile(b.bgeWeightsPath, filepath.Join(b.dataDir, ModelFile)); err != nil {
			return fmt.Errorf("copy %s: %w", ModelFile, err)
		}
		if err := copyFile(b.bgeVocabPath, filepath.Join(b.dataDir, VocabFile)); err != nil {
			return fmt.Errorf("copy %s: %w", VocabFile, err)
		}
	}

	// 5. 写 schema.json。MaxEmbedRunes 写入"实际生效值" (0 → 默认), 保证
	// 将来默认值变化时, 旧索引加载仍用当初构建时的截断长度。
	schemaToSave := *b.schema
	schemaToSave.MaxEmbedRunes = effectiveMaxRunes(b.schema.MaxEmbedRunes)
	if err := schemaToSave.SaveToFile(filepath.Join(b.dataDir, "schema.json")); err != nil {
		return fmt.Errorf("write schema.json: %w", err)
	}
	return nil
}

func (b *Builder) progress(stage string, cur, total int) {
	if b.OnProgress != nil {
		b.OnProgress(stage, cur, total)
	}
}

// embedField: 计算字段 fi 的 embedding 到 emb (n*dim), 写 embPath。
// 若启用 embCacheDir, 先按「字段内容哈希」命中缓存, 只并行编码 miss 的文档,
// 并把当前语料的所有哈希/向量写回缓存 (顺带清掉已删除文档的孤儿条目)。
func (b *Builder) embedField(fi int, emb []float32, embPath string, embDim int) error {
	name := b.schema.Fields[fi].Name
	n := len(b.items)
	workers := b.workerCount()

	// 实际截断长度以 bge 为准 (决定 embedding 输入, 也决定缓存 key)
	maxRunes := DefaultMaxEmbedRunes
	if b.bge != nil {
		maxRunes = b.bge.maxRunes
	}

	var cachePath string
	var cache map[uint64][]float32
	if b.embCacheDir != "" {
		cachePath = filepath.Join(b.embCacheDir, fmt.Sprintf("emb_%s.cache", name))
		c, err := loadEmbCache(cachePath, embDim)
		if err != nil {
			c = map[uint64][]float32{} // 缓存文件损坏 → 当空, 全量重算
		}
		cache = c
	}

	// 命中缓存, 收集未命中
	hashes := make([]uint64, n)
	missing := make([]int, 0, n)
	for i, it := range b.items {
		hashes[i] = hashTerm(truncateRunes(it.Fields[fi], maxRunes))
		if cache != nil {
			if v, ok := cache[hashes[i]]; ok && len(v) == embDim {
				copy(emb[i*embDim:(i+1)*embDim], v)
				continue
			}
		}
		missing = append(missing, i)
	}

	if len(missing) > 0 {
		b.progress("embed_"+name, 0, n)
		var done int64
		var wg sync.WaitGroup
		workCh := make(chan int, workers*2)
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := range workCh {
					vec := b.bge.EmbedFresh(b.items[i].Fields[fi])
					copy(emb[i*embDim:(i+1)*embDim], vec)
					if atomic.AddInt64(&done, 1)%200 == 0 {
						b.progress("embed_"+name, int(atomic.LoadInt64(&done)), n)
					}
				}
			}()
		}
		for _, i := range missing {
			workCh <- i
		}
		close(workCh)
		wg.Wait()
	}
	b.progress("embed_"+name, n, n)

	if err := writeEmbBin(embPath, emb, n, embDim); err != nil {
		return fmt.Errorf("write emb_%s.bin: %w", name, err)
	}

	// 回写缓存: 只含当前语料的哈希 (孤儿自动清除)。失败不致命 (索引已正确写盘)。
	if cachePath != "" {
		nc := make(map[uint64][]float32, n)
		for i := 0; i < n; i++ {
			nc[hashes[i]] = append([]float32(nil), emb[i*embDim:(i+1)*embDim]...)
		}
		_ = saveEmbCache(cachePath, embDim, nc)
	}
	return nil
}

// writeItemsBin: 写 items.bin
//
// header:
//
//	magic "SKIH" u32 | N u32 | field_count u32 | path_buf_size u32
//	tags_buf_size u32 | field_buf_sizes[field_count] u32 | source_buf_size u32
//
// per-item:
//
//	field_offsets[field_count] u32 | path_offset u32 | tags_offset u32
//	source_len u8 | source bytes
//
// buffers: field_buf_0 .. field_buf_{k-1} | paths_buf | tags_buf | sources_buf
func writeItemsBin(b *Builder, path string,
	itemFieldOffs [][]uint32, itemPathOffs, itemTagsOffs, itemSourceOffs []uint32,
	fieldBuf [][]byte, pathsBuf, tagsBuf, sourcesBuf []byte,
) error {
	n := len(itemFieldOffs)
	fieldCount := len(fieldBuf)
	if fieldCount != len(b.schema.Fields) {
		return fmt.Errorf("field count mismatch")
	}

	headerSize := 4 + 4 + 4 + 4 + 4 + 4*fieldCount + 4
	totalSize := headerSize
	perItem := 4*fieldCount + 4 + 4 + 1
	totalSize += perItem * n
	for _, it := range b.items {
		totalSize += len(it.Source)
	}
	for _, buf := range fieldBuf {
		totalSize += len(buf)
	}
	totalSize += len(pathsBuf) + len(tagsBuf) + len(sourcesBuf)

	data := make([]byte, 0, totalSize)
	data = appendUint32(data, ItemsMagic)
	data = appendUint32(data, uint32(n))
	data = appendUint32(data, uint32(fieldCount))
	data = appendUint32(data, uint32(len(pathsBuf)))
	data = appendUint32(data, uint32(len(tagsBuf)))
	for i := 0; i < fieldCount; i++ {
		data = appendUint32(data, uint32(len(fieldBuf[i])))
	}
	data = appendUint32(data, uint32(len(sourcesBuf)))

	for i := 0; i < n; i++ {
		for fi := 0; fi < fieldCount; fi++ {
			data = appendUint32(data, itemFieldOffs[i][fi])
		}
		data = appendUint32(data, itemPathOffs[i])
		data = appendUint32(data, itemTagsOffs[i])
		src := b.items[i].Source
		if len(src) > 255 {
			src = src[:255]
		}
		data = append(data, uint8(len(src)))
		data = append(data, src...)
	}

	for i := 0; i < fieldCount; i++ {
		data = append(data, fieldBuf[i]...)
	}
	data = append(data, pathsBuf...)
	data = append(data, tagsBuf...)
	data = append(data, sourcesBuf...)

	return os.WriteFile(path, data, 0644)
}

func appendUint32(b []byte, v uint32) []byte {
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], v)
	return append(b, buf[:]...)
}

// copyFile: 复制文件 (src==dst 时跳过, 避免 os.Create 截断自己)
func copyFile(src, dst string) error {
	if samePath(src, dst) {
		return nil
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, in)
	return err
}

// samePath: 判断两个路径是否指向同一文件 (abs 相等即可; 硬链接场景
// 在满足 abs 相等时同样会被正确跳过)
func samePath(a, b string) bool {
	absA, errA := filepath.Abs(a)
	absB, errB := filepath.Abs(b)
	if errA != nil || errB != nil {
		return a == b
	}
	return absA == absB
}
