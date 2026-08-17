package gs

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func TestEmbCacheRoundtrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "emb_body.cache")
	m := map[uint64][]float32{
		1: {1.5, 2.5},
		2: {3.5, 4.5},
	}
	if err := saveEmbCache(path, 2, m); err != nil {
		t.Fatal(err)
	}
	got, err := loadEmbCache(path, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[1][0] != 1.5 || got[2][1] != 4.5 {
		t.Fatalf("roundtrip got %v", got)
	}
	// dim 不匹配 → 视为失效 (空缓存)
	got2, err := loadEmbCache(path, 3)
	if err != nil || len(got2) != 0 {
		t.Fatalf("dim mismatch: got %v err=%v, want empty", got2, err)
	}
}

// TestEmbedCacheIncremental: 用一个最小 BGE 模型验证「新增文档只重算新增部分」
func TestEmbedCacheIncremental(t *testing.T) {
	base := t.TempDir()
	modelPath := filepath.Join(base, "model.safetensors")
	vocabPath := filepath.Join(base, "vocab.txt")
	cacheDir := filepath.Join(base, "cache")
	indexDir := filepath.Join(base, "idx")
	cachePath := filepath.Join(cacheDir, "emb_body.cache")

	// 最小 BGE 模型 (H=8, 4 层)
	const V, H, P, TV, L, F = 32, 8, 4, 2, 4, 16
	var tensors []tens
	add := func(name string, shape []int, data []float32) { tensors = append(tensors, tens{name, shape, data}) }
	add("embeddings.word_embeddings.weight", []int{V, H}, zeros(V*H))
	add("embeddings.position_embeddings.weight", []int{P, H}, zeros(P*H))
	add("embeddings.token_type_embeddings.weight", []int{TV, H}, zeros(TV*H))
	add("embeddings.LayerNorm.weight", []int{H}, zeros(H))
	add("embeddings.LayerNorm.bias", []int{H}, zeros(H))
	for i := 0; i < L; i++ {
		pre := "encoder.layer." + strconv.Itoa(i) + "."
		add(pre+"attention.self.query.weight", []int{H, H}, zeros(H*H))
		add(pre+"attention.self.query.bias", []int{H}, zeros(H))
		add(pre+"attention.self.key.weight", []int{H, H}, zeros(H*H))
		add(pre+"attention.self.key.bias", []int{H}, zeros(H))
		add(pre+"attention.self.value.weight", []int{H, H}, zeros(H*H))
		add(pre+"attention.self.value.bias", []int{H}, zeros(H))
		add(pre+"attention.output.dense.weight", []int{H, H}, zeros(H*H))
		add(pre+"attention.output.dense.bias", []int{H}, zeros(H))
		add(pre+"attention.output.LayerNorm.weight", []int{H}, zeros(H))
		add(pre+"attention.output.LayerNorm.bias", []int{H}, zeros(H))
		add(pre+"intermediate.dense.weight", []int{F, H}, zeros(F*H))
		add(pre+"intermediate.dense.bias", []int{F}, zeros(F))
		add(pre+"output.dense.weight", []int{H, F}, zeros(H*F))
		add(pre+"output.dense.bias", []int{H}, zeros(H))
		add(pre+"output.LayerNorm.weight", []int{H}, zeros(H))
		add(pre+"output.LayerNorm.bias", []int{H}, zeros(H))
	}
	if err := os.WriteFile(modelPath, encodeSafetensors(tensors), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(vocabPath, []byte("[PAD]\n[CLS]\n[SEP]\n[UNK]\na\n##b\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	schema := &Schema{Name: "t", Fields: []Field{
		{Name: "body", Type: FieldLongText, Embeddable: true, Searchable: true, Display: true, FieldWeight: 1.0},
	}}

	build := func(docs map[string]string) {
		t.Helper()
		b, err := NewBuilder(schema, indexDir, WithBGEPaths(modelPath, vocabPath), WithEmbedCache(cacheDir))
		if err != nil {
			t.Fatalf("NewBuilder: %v", err)
		}
		defer b.Close()
		for id, body := range docs {
			if err := b.Add(Item{Path: id, Fields: map[string]string{"body": body}}); err != nil {
				t.Fatal(err)
			}
		}
		if err := b.Build(); err != nil {
			t.Fatalf("Build: %v", err)
		}
	}

	build(map[string]string{"d1": "hello world", "d2": "foo bar"})
	c1, err := loadEmbCache(cachePath, H)
	if err != nil || len(c1) != 2 {
		t.Fatalf("after first build: cache len=%d err=%v, want 2", len(c1), err)
	}

	// 新增一个文档, 重建 → 缓存应增到 3 条 (前两条复用)
	build(map[string]string{"d1": "hello world", "d2": "foo bar", "d3": "baz qux"})
	c2, err := loadEmbCache(cachePath, H)
	if err != nil || len(c2) != 3 {
		t.Fatalf("after add doc: cache len=%d err=%v, want 3", len(c2), err)
	}

	// 删除一个文档, 重建 → 缓存回到 2 条 (孤儿被清)
	build(map[string]string{"d1": "hello world", "d3": "baz qux"})
	c3, err := loadEmbCache(cachePath, H)
	if err != nil || len(c3) != 2 {
		t.Fatalf("after delete doc: cache len=%d err=%v, want 2", len(c3), err)
	}
}
