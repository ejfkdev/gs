package gs

import (
	"testing"
)

// buildIndexDir: 构建一个最小 BM25 索引到指定目录, 返回目录路径
func buildIndexDir(t *testing.T, name string) string {
	t.Helper()
	dir := t.TempDir()
	schema := &Schema{
		Name: name,
		Fields: []Field{
			{Name: "title", Type: FieldText, Searchable: true, FieldWeight: 5.0, Display: true},
			{Name: "body", Type: FieldLongText, Searchable: true, FieldWeight: 1.0, Display: true, Snippet: true},
		},
	}
	b, err := NewBuilder(schema, dir)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	if err := b.Add(Item{Path: "a.md", Fields: map[string]string{"title": "Alpha", "body": "alpha note"}}); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if err := b.Build(); err != nil {
		t.Fatalf("Build: %v", err)
	}
	return dir
}

func TestRegistryReuse(t *testing.T) {
	dir := buildIndexDir(t, "r1")
	reg := NewRegistry()

	eng1, err := reg.Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	eng2, err := reg.Open(dir) // 同一目录再次打开应复用
	if err != nil {
		t.Fatalf("Open 2nd: %v", err)
	}
	if eng1 != eng2 {
		t.Fatal("Open same dir should return the same *Engine (cached)")
	}
	if reg.Len() != 1 {
		t.Fatalf("Len() = %d, want 1", reg.Len())
	}

	reg.Release(eng1)
	if reg.Len() != 1 {
		t.Fatalf("after one Release: Len() = %d, want 1 (refcount>0)", reg.Len())
	}
	reg.Release(eng2)
	if reg.Len() != 0 {
		t.Fatalf("after refcount drops to 0: Len() = %d, want 0", reg.Len())
	}
}

func TestRegistryMultipleInstances(t *testing.T) {
	reg := NewRegistry()
	dirA := buildIndexDir(t, "a")
	dirB := buildIndexDir(t, "b")

	engA, err := reg.Open(dirA)
	if err != nil {
		t.Fatalf("Open A: %v", err)
	}
	engB, err := reg.Open(dirB)
	if err != nil {
		t.Fatalf("Open B: %v", err)
	}
	if engA == engB {
		t.Fatal("different dirs must yield distinct engines")
	}
	if reg.Len() != 2 {
		t.Fatalf("Len() = %d, want 2", reg.Len())
	}

	// 两个实例同时可用
	if engA.Schema().Name != "a" || engB.Schema().Name != "b" {
		t.Fatalf("schemas got mixed up: %q vs %q", engA.Schema().Name, engB.Schema().Name)
	}

	reg.CloseAll()
	if reg.Len() != 0 {
		t.Fatalf("after CloseAll: Len() = %d, want 0", reg.Len())
	}
}

func TestRegistryRemove(t *testing.T) {
	dir := buildIndexDir(t, "r3")
	reg := NewRegistry()

	eng, _ := reg.Open(dir)
	reg.Remove(dir) // 即使还持有引用也强制回收
	if reg.Len() != 0 {
		t.Fatalf("after Remove: Len() = %d, want 0", reg.Len())
	}

	// Remove 后重新 Open 应得到全新实例 (重新解析)
	eng2, err := reg.Open(dir)
	if err != nil {
		t.Fatalf("re-Open: %v", err)
	}
	if eng2 == eng {
		t.Fatal("Open after Remove should return a fresh engine")
	}
	reg.CloseAll()
}
