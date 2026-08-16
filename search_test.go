package gs

import (
	"context"
	"reflect"
	"testing"
)

// testSchema: 纯 BM25 测试 schema (无 BGE, 快速)
var testSchema = &Schema{
	Name: "test",
	Fields: []Field{
		{Name: "name", Type: FieldText, Searchable: true, FieldWeight: 5.0, Display: true},
		{Name: "body", Type: FieldLongText, Searchable: true, FieldWeight: 1.0, Display: true, Snippet: true},
	},
}

func buildTestIndex(t *testing.T, docs []Item) *Engine {
	t.Helper()
	dir := t.TempDir()
	b, err := NewBuilder(testSchema, dir)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	for _, d := range docs {
		if err := b.Add(d); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}
	if err := b.Build(); err != nil {
		t.Fatalf("Build: %v", err)
	}
	eng, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Cleanup(func() { eng.Close() })
	return eng
}

func TestBuildSearchRoundtrip(t *testing.T) {
	eng := buildTestIndex(t, []Item{
		{ID: "a.md", Path: "a.md", Fields: map[string]string{"name": "Nginx 使用指南", "body": "Nginx 反向代理的配置与部署说明, 示例见附录"}},
		{ID: "b.md", Path: "b.md", Fields: map[string]string{"name": "Nginx 配置", "body": "nginx 反向代理配置示例"}},
		{ID: "c.md", Path: "c.md", Fields: map[string]string{"name": "Markdown 渲染", "body": "前端渲染 markdown 的实现方式"}},
	})

	hits, err := eng.Search(context.Background(), SearchOptions{Query: "nginx 部署", TopK: 5})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) == 0 || hits[0].ID != "a.md" {
		t.Fatalf("top hit = %+v, want a.md", hits)
	}
	if hits[0].Snippet == "" {
		t.Error("expected non-empty snippet")
	}

	// CJK bigram 召回
	hits, _ = eng.Search(context.Background(), SearchOptions{Query: "markdown 渲染", TopK: 5})
	if len(hits) == 0 || hits[0].ID != "c.md" {
		t.Fatalf("CJK bigram query top hit = %+v, want c.md", hits)
	}

	// 空查询 → 空结果
	hits, _ = eng.Search(context.Background(), SearchOptions{Query: "", TopK: 5})
	if hits != nil {
		t.Fatalf("empty query should return nil, got %v", hits)
	}
}

func TestSearchStrictBoost(t *testing.T) {
	eng := buildTestIndex(t, []Item{
		{ID: "app.md", Path: "app.md", Fields: map[string]string{"name": "App", "body": "描述一"}},
		{ID: "net.md", Path: "net.md", Fields: map[string]string{"name": "网络笔记", "body": "内网段 192.168.1.1 的部署记录"}},
	})
	hits, err := eng.Search(context.Background(), SearchOptions{Query: "192.168.1.1", TopK: 5})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) == 0 || hits[0].Path != "net.md" {
		t.Fatalf("strict boost top hit = %+v, want net.md", hits)
	}
}

func TestSearchCacheIsolation(t *testing.T) {
	eng := buildTestIndex(t, []Item{
		{ID: "d1", Path: "a.md", Fields: map[string]string{"name": "Alpha", "body": "one two three four five six seven eight nine ten"}},
		{ID: "d2", Path: "b.md", Fields: map[string]string{"name": "Beta", "body": "alpha note"}},
	})

	ctx := context.Background()

	// 相同 query 不同 TopK 应返回不同数量 (旧实现只按 query 缓存, 这里回归测试)
	h1, _ := eng.Search(ctx, SearchOptions{Query: "alpha note", TopK: 5})
	h2, _ := eng.Search(ctx, SearchOptions{Query: "alpha note", TopK: 1})
	if len(h2) != 1 {
		t.Fatalf("TopK=1 returned %d hits (cache key misses TopK?), want 1", len(h2))
	}
	if len(h1) < len(h2) {
		t.Fatalf("TopK=5 returned %d hits, want >= %d", len(h1), len(h2))
	}

	// 修改返回的 hit 不应污染缓存
	h2[0].ID = "tainted"
	h3, _ := eng.Search(ctx, SearchOptions{Query: "alpha note", TopK: 1})
	if h3[0].ID == "tainted" {
		t.Fatal("cache returned mutated hit — Get must return a copy")
	}
}

func TestFastSearch(t *testing.T) {
	eng := buildTestIndex(t, []Item{
		{ID: "d1", Path: "a.md", Fields: map[string]string{"name": "Nginx 使用指南", "body": "Nginx 反向代理配置说明"}},
		{ID: "d2", Path: "b.md", Fields: map[string]string{"name": "Nginx", "body": "web server"}},
	})
	res := eng.FastSearch("nginx 配置与部署的说明文档", 5)
	if len(res) == 0 || res[0].Name == "" {
		t.Fatalf("FastSearch: got %+v", res)
	}
	if !reflect.DeepEqual(eng.Schema().Fields, testSchema.Fields) {
		t.Fatal("schema mutated unexpectedly")
	}
}

func TestExtractSnippets(t *testing.T) {
	text := "这里是对 Nginx 配置说明, 包括反向代理与缓存策略的用法, 内容较长"
	s := ExtractSnippets(text, "Nginx 配置", 2, 40, 60)
	if s == "" {
		t.Fatal("expected non-empty snippet")
	}
}

func TestStrictTokens(t *testing.T) {
	toks := ExtractStrictTokens("访问 192.168.1.1 和 example.com 以及 a@b.co")
	types := map[string]bool{}
	for _, tok := range toks {
		types[tok.Type] = true
	}
	for _, want := range []string{"ip", "domain", "email"} {
		if !types[want] {
			t.Errorf("missing strict token type %q in %+v", want, toks)
		}
	}
}

// TestSearchStrictNoTokens: 回归 — SearchOptions.Strict=true 且 query 没有
// strict 格式 token (IP/域名/哈希/URL 等) 时, strictBoosts 之前返回 nil,
// 导致 strictN[i] 越界 panic。
func TestSearchStrictNoTokens(t *testing.T) {
	eng := buildTestIndex(t, []Item{
		{ID: "a.md", Path: "a.md", Fields: map[string]string{"name": "Nginx 配置", "body": "反向代理与缓存"}},
	})
	hits, err := eng.Search(context.Background(), SearchOptions{Query: "nginx 配置", TopK: 5, Strict: true})
	if err != nil {
		t.Fatalf("Search(Strict=true, no strict token): %v", err)
	}
	if len(hits) == 0 || hits[0].Path != "a.md" {
		t.Fatalf("unexpected hits: %+v", hits)
	}
}

// TestComputeStrictBoostNoTokens: strictBoosts 在无 token 时应返回 n 个 0,
// 而不是 nil。
func TestComputeStrictBoostNoTokens(t *testing.T) {
	boosts, types := computeStrictBoost("普通文本 无任何特殊格式", [][]StrictField{{}}, 3)
	if len(boosts) != 3 {
		t.Fatalf("boosts len = %d, want 3", len(boosts))
	}
	for _, v := range boosts {
		if v != 0 {
			t.Fatalf("boosts should be all zero, got %v", boosts)
		}
	}
	if types != nil {
		t.Fatalf("types = %v, want nil", types)
	}
}

// TestFullyCustomSchema: 完全自定义字段 (无 name/tags), 验证库不依赖任何内置字段名
func TestFullyCustomSchema(t *testing.T) {
	schema := &Schema{
		Name:      "inventory",
		NameField: "sku", // 精确/前缀匹配字段自定义为 sku
		Fields: []Field{
			{Name: "sku", Type: FieldText, Searchable: true, FieldWeight: 5.0, Display: true},
			{Name: "summary", Type: FieldLongText, Searchable: true, FieldWeight: 1.0, Display: true, Snippet: true},
			{Name: "labels", Type: FieldTags, Searchable: true, FieldWeight: 3.0, Display: true},
		},
	}

	dir := t.TempDir()
	b, err := NewBuilder(schema, dir)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	if err := b.Add(Item{
		Path:   "sku/123.md",
		Source: "inv",
		Tags:   []string{"premium", "sale"},
		Fields: map[string]string{
			"sku":     "widget-123",
			"summary": "红色外壳的桌面小部件, 带无线充电功能",
		},
	}); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if err := b.Build(); err != nil {
		t.Fatalf("Build: %v", err)
	}

	eng, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer eng.Close()

	if got := eng.Schema().PrimaryField(); got != "sku" {
		t.Fatalf("PrimaryField() = %q, want sku", got)
	}

	// 精确/前缀匹配走自定义 NameField
	hits, _ := eng.Search(context.Background(), SearchOptions{Query: "widget-123", TopK: 5})
	if len(hits) == 0 || hits[0].Path != "sku/123.md" {
		t.Fatalf("sku query top hit = %+v, want sku/123.md", hits)
	}

	// Item.Tags 被合并进自定义的 FieldTags 字段 "labels", 且可被搜索到
	hits, _ = eng.Search(context.Background(), SearchOptions{Query: "premium", TopK: 5})
	if len(hits) == 0 || hits[0].Path != "sku/123.md" {
		t.Fatalf("tag query top hit = %+v, want sku/123.md", hits)
	}
	if len(hits[0].Tags) != 2 {
		t.Fatalf("Hit.Tags = %v, want 2 items", hits[0].Tags)
	}
}
