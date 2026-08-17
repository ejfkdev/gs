package gs

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"testing"
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
		"a":    map[string]interface{}{"b": []interface{}{"x", "y", "z"}},
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

func TestMatchGlob(t *testing.T) {
	cases := []struct {
		pattern, name string
		want          bool
	}{
		{"**/*.json", "a/b/c.json", true},
		{"*.json", "a.json", true},
		{"*.json", "a/b.json", false},
		{"docs/**/*.md", "docs/x/y.md", true},
		{"docs/*.md", "docs/x.md", true},
		{"docs/**/*.md", "README.md", false},
	}
	for _, c := range cases {
		if got := MatchGlob(c.pattern, c.name); got != c.want {
			t.Errorf("MatchGlob(%q, %q) = %v, want %v", c.pattern, c.name, got, c.want)
		}
	}
}

func TestExtractFormats(t *testing.T) {
	tags := map[string]bool{"tags": true, "keys": true}
	noTags := map[string]bool{}

	cases := []struct {
		format  string
		data    string
		mapping map[string]string
	}{
		{"json", `{"title":"T","body":"B","keys":["a","b"]}`, map[string]string{"title": "title", "body": "body", "tags": "keys"}},
		{"yaml", "title: T\nbody: B\nkeys:\n  - a\n  - b\n", map[string]string{"title": "title", "body": "body", "tags": "keys"}},
	}
	for _, c := range cases {
		p := filepath.Join(t.TempDir(), "f")
		writeFile(t, p, c.data)
		items, err := extractSource(SourceConfig{Format: c.format, Mapping: c.mapping}, tags, p, "f")
		if err != nil {
			t.Fatalf("%s: %v", c.format, err)
		}
		if len(items) != 1 || items[0].Fields["title"] != "T" || !reflect.DeepEqual(items[0].Tags, []string{"a", "b"}) {
			t.Fatalf("%s: items = %+v", c.format, items)
		}
	}

	// csv (多行)
	csvPath := filepath.Join(t.TempDir(), "c.csv")
	writeFile(t, csvPath, "title,body\nT1,B1\nT2,B2\n")
	items, err := extractSource(SourceConfig{Format: "csv", Mapping: map[string]string{"title": "title", "body": "body"}}, noTags, csvPath, "c.csv")
	if err != nil || len(items) != 2 || items[0].Fields["title"] != "T1" {
		t.Fatalf("csv items = %+v, err=%v", items, err)
	}

	// ndjson
	ndPath := filepath.Join(t.TempDir(), "d.ndjson")
	writeFile(t, ndPath, "{\"title\":\"N1\"}\n{\"title\":\"N2\"}\n")
	items, err = extractSource(SourceConfig{Format: "ndjson", Mapping: map[string]string{"title": "title"}}, noTags, ndPath, "d.ndjson")
	if err != nil || len(items) != 2 || items[0].Fields["title"] != "N1" || items[1].Fields["title"] != "N2" {
		t.Fatalf("ndjson items = %+v, err=%v", items, err)
	}
}

func TestIndexConfigBuild(t *testing.T) {
	tmp := t.TempDir()
	src := filepath.Join(tmp, "data")
	writeFile(t, filepath.Join(src, "a.json"), `{"title":"Log4j 配置","content":"日志配置说明","keys":["log","config"]}`)
	writeFile(t, filepath.Join(src, "b.json"), `{"title":"Nginx 反向代理","content":"代理与缓存","keys":["web"]}`)
	writeFile(t, filepath.Join(src, "items.csv"), "title,content\nCSV 条目一,内容一\nCSV 条目二,内容二\n")

	cfgYAML := `schema:
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
	cfg, err := ParseIndexConfig([]byte(cfgYAML))
	if err != nil {
		t.Fatalf("ParseIndexConfig: %v", err)
	}
	outDir := filepath.Join(tmp, "out")

	stats, err := cfg.Build(outDir)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if stats.Items != 4 { // 2 json + 2 csv
		t.Fatalf("stats.Items = %d, want 4", stats.Items)
	}

	eng, err := Load(outDir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer eng.Close()

	hits, _ := eng.Search(context.Background(), SearchOptions{Query: "nginx 反向代理", TopK: 5})
	if len(hits) == 0 || hits[0].Fields["title"] != "Nginx 反向代理" {
		t.Fatalf("top hit = %+v", hits)
	}
	hits, _ = eng.Search(context.Background(), SearchOptions{Query: "CSV 条目一", TopK: 5})
	if len(hits) == 0 {
		t.Fatalf("csv item not indexed")
	}
}
