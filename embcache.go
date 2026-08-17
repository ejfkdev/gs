// embcache.go - 持久化的 per-field embedding 缓存
//
// 以「字段内容哈希」为 key 存每个字段的向量, 让重建只算新增/变更的文档。
// 与按位置(下标)复用的 emb_<field>.bin 不同, 哈希 key 与文档顺序/增删无关,
// 所以新增、删除、改动单个文档都不会让其余文档失效。
//
// 代价: 缓存磁盘占用 ≈ embedding 本身 (每个字段一份 .cache)。不需要增量时
// 不传 WithEmbedCache / IndexWithEmbedCache / --emb-cache。

package gs

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
)

// embCacheMagic: 缓存文件魔数 "ECHC" (little-endian)
const embCacheMagic = 0x43484345

// loadEmbCache: 读缓存 (文件不存在返回空; dim 不匹配视为失效返回空)
func loadEmbCache(path string, dim int) (map[uint64][]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return map[uint64][]float32{}, nil
		}
		return nil, err
	}
	if len(data) < 16 {
		return nil, fmt.Errorf("emb cache %s: too short", path)
	}
	pos := 0
	if binary.LittleEndian.Uint32(data[pos:]) != embCacheMagic {
		return nil, fmt.Errorf("emb cache %s: bad magic", path)
	}
	pos += 4
	if int(binary.LittleEndian.Uint32(data[pos:])) != dim {
		return map[uint64][]float32{}, nil // dim 变化 → 失效
	}
	pos += 4
	count := binary.LittleEndian.Uint64(data[pos:])
	pos += 8

	m := make(map[uint64][]float32, count)
	for i := uint64(0); i < count; i++ {
		if pos+8+dim*4 > len(data) {
			return nil, fmt.Errorf("emb cache %s: truncated", path)
		}
		h := binary.LittleEndian.Uint64(data[pos:])
		pos += 8
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = math.Float32frombits(binary.LittleEndian.Uint32(data[pos:]))
			pos += 4
		}
		m[h] = vec
	}
	return m, nil
}

// saveEmbCache: 原子写缓存 (临时文件 + rename)
func saveEmbCache(path string, dim int, m map[uint64][]float32) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	// 排序 key, 保证文件字节确定性
	keys := make([]uint64, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })

	size := 4 + 4 + 8 + len(keys)*(8+dim*4)
	buf := make([]byte, 0, size)
	buf = binary.LittleEndian.AppendUint32(buf, embCacheMagic)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(dim))
	buf = binary.LittleEndian.AppendUint64(buf, uint64(len(keys)))
	for _, k := range keys {
		buf = binary.LittleEndian.AppendUint64(buf, k)
		for _, v := range m[k] {
			buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(v))
		}
	}

	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, buf, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
