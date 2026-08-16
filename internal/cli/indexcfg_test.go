package cli

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/ejfkdev/gs"
)

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestGetPath(t *testing.T) {
	root := map[string]interface{}{
		"a": map[string]interface{}{
			"b": []interface{}{"x", "y", "z"},
		},
		"name": "n1",
	}
	cases := []struct {
		path string
		want interface{}
		ok   bool
	}{
		{"name", "n1", true},
		{"a.b.0", "x", true},
		{"a.b[1]", "y", true},
		{"missing", nil, false},
		{"a.missing", nil, false},
	}
	for _, c := range cases {
		got, ok := getPath(root, c.path)
		if ok != c.ok || !reflect.DeepEqual(got, c.want) {
			t.Errorf("getPath(%q) = (%v, %v), want (%v, %v)", c.path, got, ok, c.want, c.ok)
		}
	}
}

func TestExtractFormats(t *testing.T) {
	tags := map[string]bool{"tags": true, "keys": true}
	noTags := map[string]bool{}
	var (
		items []gs.Item
		err   error
	)

	// json + yaml
	fmTests := []struct {
		format  string
		data    string
		mapping map[string]string
	}{
		{"json", `{"title":"T","body":"B","keys":["a","b"]}`, map[string]string{"title": "title", "body": "body", "tags": "keys"}},
		{"yaml", "title: T\nbody: B\nkeys:\n  - a\n  - b\n", map[string]string{"title": "title", "body": "body", "tags": "keys"}},
	}
	for _, ft := range fmTests {
		p := filepath.Join(t.TempDir(), "f")
		writeFile(t, p, ft.data)
		items, err = extractSourceFile(SourceConfig{Format: ft.format, Mapping: ft.mapping}, tags, p, "f")
		if err != nil {
			t.Fatalf("%s: %v", ft.format, err)
		}
		if len(items) != 1 {
			t.Fatalf("%s: got %d items, want 1", ft.format, len(items))
		}
		got := items[0]
		if got.Fields["title"] != "T" || got.Fields["body"] != "B" {
			t.Fatalf("%s: fields = %v", ft.format, got.Fields)
		}
		if !reflect.DeepEqual(got.Tags, []string{"a", "b"}) {
			t.Fatalf("%s: tags = %v, want [a b]", ft.format, got.Tags)
		}
	}

	// frontmatter
	fm := "---\nname: SK\ncategory: cat\n---\n正文内容"
	skPath := filepath.Join(t.TempDir(), "sk")
	writeFile(t, skPath, fm)
	items, err = extractSourceFile(SourceConfig{Format: "frontmatter", Mapping: map[string]string{"title": "name", "body": "__body__", "cat": "category"}}, noTags, skPath, "sk")
	if err != nil {
		t.Fatal(err)
	}
	if items[0].Fields["title"] != "SK" || items[0].Fields["cat"] != "cat" || items[0].Fields["body"] != "正文内容" {
		t.Fatalf("frontmatter fields = %v", items[0].Fields)
	}

	// csv (多行)
	csv := "title,body\nT1,B1\nT2,B2\n"
	csvPath := filepath.Join(t.TempDir(), "c.csv")
	writeFile(t, csvPath, csv)
	items, err = extractSourceFile(SourceConfig{Format: "csv", Mapping: map[string]string{"title": "title", "body": "body"}}, noTags, csvPath, "c.csv")
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 2 || items[0].Fields["title"] != "T1" || items[1].Fields["title"] != "T2" {
		t.Fatalf("csv items = %+v", items)
	}
}

func TestRunConfigBuild(t *testing.T) {
	tmp := t.TempDir()
	src := filepath.Join(tmp, "data")
	writeFile(t, filepath.Join(src, "a.json"), `{"title":"Log4j 配置","content":"日志配置说明","keys":["log","config"]}`)
	writeFile(t, filepath.Join(src, "b.json"), `{"title":"Nginx 反向代理","content":"代理与缓存","keys":["web"]}`)
	writeFile(t, filepath.Join(src, "items.csv"), "title,content\nCSV 条目一,内容一\nCSV 条目二,内容二\n")

	cfg := `schema:
  name: testidx
  name_field: title
  fields:
    - {name: title,   type: text,     searchable: true, embeddable: false, weight: 5.0, display: true}
    - {name: content, type: longtext, searchable: true, embeddable: false, weight: 1.0, display: true, snippet: true}
    - {name: keys,    type: tags,     searchable: true, embeddable: false, weight: 3.0, display: true}
sources:
  - dir: ` + src + `
    include: "*.json"
    format: json
    mapping: {title: title, content: content, keys: keys}
  - dir: ` + src + `
    include: "*.csv"
    format: csv
    mapping: {title: title, content: content}
`
	cfgPath := filepath.Join(tmp, "index.yaml")
	writeFile(t, cfgPath, cfg)
	outDir := filepath.Join(tmp, "out")

	if err := (&buildOptions{config: cfgPath, output: outDir}).runConfig(io.Discard); err != nil {
		t.Fatalf("runConfig: %v", err)
	}

	eng, err := gs.Load(outDir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer eng.Close()

	hits, err := eng.Search(context.Background(), gs.SearchOptions{Query: "nginx 反向代理", TopK: 5})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) == 0 || hits[0].Fields["title"] != "Nginx 反向代理" {
		t.Fatalf("top hit = %+v", hits)
	}
	// csv 条目也应被索引
	hits, _ = eng.Search(context.Background(), gs.SearchOptions{Query: "CSV 条目一", TopK: 5})
	if len(hits) == 0 {
		t.Fatalf("csv item not indexed: %+v", hits)
	}
}
