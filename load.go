// load.go - 从 dataDir 加载预编译的索引
// 读 schema.json + items.bin + emb_<field>.bin + model.safetensors + vocab.txt

package gs

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

// Load: 从 dataDir 加载引擎
// 流程:
//  1. 读 schema.json → Schema
//  2. 读 items.bin → items + per-field string buffer
//  3. 读 emb_<field>.bin (每个 embeddable field 一个)
//  4. 读 model.safetensors + vocab.txt → BGE engine (固定名自动读取)
//  5. 重建 per-field inverted index
func Load(dataDir string) (*Engine, error) {
	// 1. Schema
	schemaPath := filepath.Join(dataDir, "schema.json")
	data, err := os.ReadFile(schemaPath)
	if err != nil {
		return nil, fmt.Errorf("read schema.json: %w", err)
	}
	schema, err := SchemaFromJSON(data)
	if err != nil {
		return nil, fmt.Errorf("parse schema.json: %w", err)
	}
	if err := schema.Validate(); err != nil {
		return nil, fmt.Errorf("validate schema: %w", err)
	}

	// 2. items.bin
	itemsPath := filepath.Join(dataDir, "items.bin")
	items, fieldsBuf, pathsBuf, tagsBuf, sourcesBuf, err := readItemsBin(itemsPath, schema)
	if err != nil {
		return nil, fmt.Errorf("read items.bin: %w", err)
	}

	// 3+4. embeddings
	eng := NewEngine(schema)
	eng.dataDir = dataDir
	eng.items = items
	eng.fieldsBuf = fieldsBuf
	eng.pathsBuf = pathsBuf
	eng.tagsBuf = tagsBuf
	eng.sourcesBuf = sourcesBuf
	eng.n = len(items)

	// 加载 emb_<field>.bin (每个 embeddable field 一个; 搜索按字段加权使用 per-field 向量)
	for fi, f := range schema.Fields {
		if !f.Embeddable {
			continue
		}
		embPath := filepath.Join(dataDir, fmt.Sprintf("emb_%s.bin", f.Name))
		if eEmb, dim, err := readEmbBin(embPath); err == nil {
			if eng.embField == nil {
				eng.embField = make([][]float32, len(schema.Fields))
			}
			eng.embField[fi] = eEmb
			if eng.embDim == 0 {
				eng.embDim = dim
			}
		}
	}
	if eng.embDim == 0 {
		eng.embDim = DefaultEmbDim
	}

	// 5. BGE 模型/词表: 从索引目录按固定名自动读取
	weightsPath := firstExisting(filepath.Join(dataDir, ModelFile))
	vocabPath := firstExisting(filepath.Join(dataDir, VocabFile))
	if weightsPath != "" && vocabPath != "" {
		bge, err := NewBGEEngine(weightsPath, vocabPath)
		if err != nil {
			return nil, fmt.Errorf("init BGE: %w", err)
		}
		// 与索引构建时保持一致 (schema 里记录了当时用的截断长度)
		bge.maxRunes = effectiveMaxRunes(schema.MaxEmbedRunes)
		eng.bge = bge
	}

	// 6. 预计算 name 字段小写 (nameMatches / prefixMatches / rerank 共用)
	if eng.nameFieldIdx >= 0 {
		eng.nameLower = make([]string, eng.n)
		for i := 0; i < eng.n; i++ {
			eng.nameLower[i] = strings.ToLower(eng.fieldText(i, eng.nameFieldIdx))
		}
	}

	// 7. 重建 inverted index
	eng.buildInvertedIndex()

	return eng, nil
}

// readItemsBin: 读 items.bin
// header:
//
//	magic "SKIH" (4B)
//	uint32 N
//	uint32 field_count
//	uint32 path_buf_size
//	uint32 tags_buf_size
//	uint32 field_buf_sizes[field_count]
//	uint32 source_buf_size
//
// per-item:
//
//	uint32 field_offsets[field_count]
//	uint32 path_offset
//	uint32 tags_offset
//	uint8  source_len
//	bytes  source_str
//
// string buffers:
//
//	field_buf_0 .. field_buf_{field_count-1}
//	paths_buf
//	tags_buf
//	sources_buf
func readItemsBin(path string, schema *Schema) ([]*internalItem, [][]byte, []byte, []byte, []byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	if len(data) < 4 {
		return nil, nil, nil, nil, nil, fmt.Errorf("items.bin too short")
	}
	pos := 0
	magic := binary.LittleEndian.Uint32(data[pos:])
	pos += 4
	if magic != ItemsMagic {
		return nil, nil, nil, nil, nil, fmt.Errorf("bad items.bin magic: %x (want %x)", magic, ItemsMagic)
	}
	N := binary.LittleEndian.Uint32(data[pos:])
	pos += 4
	fieldCount := int(binary.LittleEndian.Uint32(data[pos:]))
	pos += 4
	if uint32(fieldCount) != uint32(len(schema.Fields)) {
		return nil, nil, nil, nil, nil, fmt.Errorf("schema field count mismatch: bin=%d, schema=%d", fieldCount, len(schema.Fields))
	}
	pathBufSize := binary.LittleEndian.Uint32(data[pos:])
	pos += 4
	tagsBufSize := binary.LittleEndian.Uint32(data[pos:])
	pos += 4
	fieldBufSizes := make([]uint32, fieldCount)
	for i := 0; i < fieldCount; i++ {
		fieldBufSizes[i] = binary.LittleEndian.Uint32(data[pos:])
		pos += 4
	}
	sourceBufSize := binary.LittleEndian.Uint32(data[pos:])
	pos += 4

	// per-item 区
	items := make([]*internalItem, N)
	for i := uint32(0); i < N; i++ {
		if pos+4*fieldCount+8+1 > len(data) {
			return nil, nil, nil, nil, nil, fmt.Errorf("items.bin truncated at item %d", i)
		}
		it := &internalItem{
			FieldOffs: make([]uint32, fieldCount),
		}
		for fi := 0; fi < fieldCount; fi++ {
			it.FieldOffs[fi] = binary.LittleEndian.Uint32(data[pos:])
			pos += 4
		}
		it.PathOff = binary.LittleEndian.Uint32(data[pos:])
		pos += 4
		it.TagsOff = binary.LittleEndian.Uint32(data[pos:])
		pos += 4
		srcLen := int(data[pos])
		pos++
		if pos+srcLen > len(data) {
			return nil, nil, nil, nil, nil, fmt.Errorf("items.bin truncated at source of item %d", i)
		}
		it.Source = string(data[pos : pos+srcLen])
		pos += srcLen
		items[i] = it
	}

	// 读 buffers
	fieldsBuf := make([][]byte, fieldCount)
	for i := 0; i < fieldCount; i++ {
		if pos+int(fieldBufSizes[i]) > len(data) {
			return nil, nil, nil, nil, nil, fmt.Errorf("items.bin truncated at field_buf %d", i)
		}
		buf := make([]byte, fieldBufSizes[i])
		copy(buf, data[pos:pos+int(fieldBufSizes[i])])
		fieldsBuf[i] = buf
		pos += int(fieldBufSizes[i])
	}
	if pos+int(pathBufSize) > len(data) {
		return nil, nil, nil, nil, nil, fmt.Errorf("items.bin truncated at paths_buf")
	}
	pathsBuf := make([]byte, pathBufSize)
	copy(pathsBuf, data[pos:pos+int(pathBufSize)])
	pos += int(pathBufSize)
	if pos+int(tagsBufSize) > len(data) {
		return nil, nil, nil, nil, nil, fmt.Errorf("items.bin truncated at tags_buf")
	}
	tagsBuf := make([]byte, tagsBufSize)
	copy(tagsBuf, data[pos:pos+int(tagsBufSize)])
	pos += int(tagsBufSize)
	if pos+int(sourceBufSize) > len(data) {
		return nil, nil, nil, nil, nil, fmt.Errorf("items.bin truncated at sources_buf")
	}
	sourcesBuf := make([]byte, sourceBufSize)
	copy(sourcesBuf, data[pos:pos+int(sourceBufSize)])
	pos += int(sourceBufSize)

	// 解析 tags 成 []string (for Hit.Tags); items.bin 不单独存 ID,
	// 加载时 ID = Path (extractor 的 ID 本来就是 relPath)
	for _, it := range items {
		tagsStr := readStr(tagsBuf, it.TagsOff)
		if tagsStr != "" {
			it.Tags = splitWords(tagsStr)
		}
		it.ID = readStr(pathsBuf, it.PathOff)
	}

	return items, fieldsBuf, pathsBuf, tagsBuf, sourcesBuf, nil
}

// splitWords: 用空白切分
func splitWords(s string) []string {
	s = trimSpaces(s)
	if s == "" {
		return nil
	}
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == ' ' || s[i] == '\t' || s[i] == '\n' {
			if i > start {
				out = append(out, s[start:i])
			}
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

func trimSpaces(s string) string {
	start := 0
	end := len(s)
	for start < end && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n') {
		start++
	}
	for end > start && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\n') {
		end--
	}
	return s[start:end]
}

// readEmbBin: 读 emb_*.bin
// header:
//
//	uint32 N
//	uint32 dim
//
// body:
//
//	float32 embs[N*dim]
func readEmbBin(path string) ([]float32, int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	if len(data) < 8 {
		return nil, 0, fmt.Errorf("emb file too short: %s", path)
	}
	N := binary.LittleEndian.Uint32(data[0:])
	dim := binary.LittleEndian.Uint32(data[4:])
	bodyLen := int(N) * int(dim) * 4
	if 8+bodyLen > len(data) {
		return nil, 0, fmt.Errorf("emb file truncated: %s (have %d, want %d)", path, len(data), 8+bodyLen)
	}
	embs := bytesToFloat32LE(data[8 : 8+bodyLen])
	return embs, int(dim), nil
}

// writeEmbBin 是 helper, 给 build.go 用
func writeEmbBin(path string, embs []float32, n, dim int) error {
	data := make([]byte, 8+len(embs)*4)
	binary.LittleEndian.PutUint32(data[0:], uint32(n))
	binary.LittleEndian.PutUint32(data[4:], uint32(dim))
	for i, v := range embs {
		bits := math.Float32bits(v)
		binary.LittleEndian.PutUint32(data[8+i*4:], bits)
	}
	return os.WriteFile(path, data, 0644)
}

// firstExisting: 返回第一个存在的路径 (都不存在返回 "")
func firstExisting(paths ...string) string {
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

// bytesToFloat32LE: 把 little-endian float32 字节流解码为 []float32。
// Load 路径里 emb_*.bin 和 safetensors 的浮点解码是纯标量循环 (约几千万
// 次), 这里按 CPU 核数分块并行, 是加载耗时的最主要优化点。
func bytesToFloat32LE(b []byte) []float32 {
	n := len(b) / 4
	out := make([]float32, n)
	if n < 16384 { // < 64KB 就串行, 避免并发开销
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
		}
		return out
	}

	workers := runtime.NumCPU()
	if workers > n/16384 {
		workers = n / 16384
	}
	if workers < 1 {
		workers = 1
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * chunk
		if start >= n {
			break
		}
		end := start + chunk
		if end > n {
			end = n
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
			}
		}(start, end)
	}
	wg.Wait()
	return out
}
