package gs

import (
	"context"
	"testing"
	"time"
)

func buildTo(t *testing.T, dir string, titles ...string) {
	t.Helper()
	schema := &Schema{
		Name: "t",
		Fields: []Field{
			{Name: "title", Type: FieldText, Searchable: true, FieldWeight: 5.0, Display: true},
		},
	}
	b, err := NewBuilder(schema, dir)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	for _, title := range titles {
		if err := b.Add(Item{Path: title, Fields: map[string]string{"title": title}}); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}
	if err := b.Build(); err != nil {
		t.Fatalf("Build: %v", err)
	}
}

func TestLiveEngineReload(t *testing.T) {
	dir := t.TempDir()
	buildTo(t, dir, "Alpha")

	l, err := OpenLive(dir, 50*time.Millisecond)
	if err != nil {
		t.Fatalf("OpenLive: %v", err)
	}
	defer l.Close()

	hits, _ := l.Search(context.Background(), SearchOptions{Query: "Alpha", TopK: 5})
	if len(hits) == 0 || hits[0].Fields["title"] != "Alpha" {
		t.Fatalf("initial search = %+v", hits)
	}

	// 重建 (新增一个文档)
	buildTo(t, dir, "Alpha", "Beta")
	time.Sleep(200 * time.Millisecond) // 等自动重载

	hits, _ = l.Search(context.Background(), SearchOptions{Query: "Beta", TopK: 5})
	if len(hits) == 0 || hits[0].Fields["title"] != "Beta" {
		t.Fatalf("after reload search = %+v, want Beta", hits)
	}
}

// TestLiveEngineCloseIdempotent: Close 可安全重入 (sync.Once 保证, 不应二次 close channel panic)
func TestLiveEngineCloseIdempotent(t *testing.T) {
	dir := t.TempDir()
	buildTo(t, dir, "Alpha")
	l, err := OpenLive(dir, time.Second)
	if err != nil {
		t.Fatalf("OpenLive: %v", err)
	}
	if err := l.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := l.Close(); err != nil { // 二次关闭应安全
		t.Fatalf("second Close: %v", err)
	}
}
