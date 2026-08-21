package xyzsvc

import (
	"context"
	"testing"

	"github.com/ejfkdev/gs"
)

// buildIndex 建一个纯 BM25 (无 BGE) 的小索引, 快速且不依赖模型文件。
func buildIndex(t *testing.T, dir string) {
	t.Helper()
	schema := &gs.Schema{
		Name: "t",
		Fields: []gs.Field{
			{Name: "name", Type: gs.FieldText, Searchable: true, FieldWeight: 5.0, Display: true},
			{Name: "body", Type: gs.FieldLongText, Searchable: true, FieldWeight: 1.0, Display: true, Snippet: true},
		},
	}
	b, err := gs.NewBuilder(schema, dir)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	defer b.Close()
	docs := []gs.Item{
		{ID: "a.md", Path: "a.md", Fields: map[string]string{"name": "Nginx 使用指南", "body": "Nginx 反向代理的配置与部署说明"}},
		{ID: "b.md", Path: "b.md", Fields: map[string]string{"name": "Markdown 渲染", "body": "前端渲染 markdown 的实现方式"}},
	}
	for _, d := range docs {
		if err := b.Add(d); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}
	if err := b.Build(); err != nil {
		t.Fatalf("Build: %v", err)
	}
}

func TestRegistryCommands(t *testing.T) {
	reg, err := Registry()
	if err != nil {
		t.Fatalf("Registry: %v", err)
	}
	names := reg.Names()
	want := []string{"fastsearch", "index", "schema", "search"}
	if len(names) != 4 || names[0] != want[0] || names[1] != want[1] || names[2] != want[2] || names[3] != want[3] {
		t.Fatalf("names = %v, want %v", names, want)
	}
}

func TestSearchAndSchemaHandlers(t *testing.T) {
	t.Cleanup(Close)
	dir := t.TempDir()
	buildIndex(t, dir)

	hits, err := searchHandler(context.Background(), &SearchArgs{Index: dir, Query: "nginx 部署", K: 5})
	if err != nil {
		t.Fatalf("searchHandler: %v", err)
	}
	if len(hits) == 0 || hits[0].Path != "a.md" {
		t.Fatalf("top hit = %+v, want a.md", hits)
	}

	schema, err := schemaHandler(context.Background(), &SchemaArgs{Index: dir})
	if err != nil {
		t.Fatalf("schemaHandler: %v", err)
	}
	if schema.Name != "t" || len(schema.Fields) != 2 || schema.Fields[0].Name != "name" {
		t.Fatalf("schema = %+v", schema)
	}
}

func TestFastSearchHandler(t *testing.T) {
	dir := t.TempDir()
	buildIndex(t, dir)

	res, err := fastSearchHandler(context.Background(), &FastSearchArgs{Index: dir, Text: "nginx 配置与部署的说明文档", K: 5})
	if err != nil {
		t.Fatalf("fastSearchHandler: %v", err)
	}
	if len(res) == 0 || res[0].Path != "a.md" {
		t.Fatalf("top = %+v, want a.md", res)
	}
}

func TestValidateFields(t *testing.T) {
	dir := t.TempDir()
	buildIndex(t, dir)
	if err := validateFields(dir, []string{"name"}); err != nil {
		t.Fatalf("validateFields(name): %v", err)
	}
	if err := validateFields(dir, []string{"nope"}); err == nil {
		t.Fatal("validateFields(unknown) should error")
	}
}

func TestEngineForCaches(t *testing.T) {
	t.Cleanup(Close)
	dir := t.TempDir()
	buildIndex(t, dir)

	e1, err := engineFor(dir)
	if err != nil {
		t.Fatalf("engineFor: %v", err)
	}
	e2, err := engineFor(dir)
	if err != nil {
		t.Fatalf("engineFor 2: %v", err)
	}
	if e1 != e2 {
		t.Fatal("engineFor should return the same cached instance")
	}
}
