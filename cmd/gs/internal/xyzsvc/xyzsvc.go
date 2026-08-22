// Package xyzsvc integrates the gs search library with xyz-go: each gs
// capability is defined once as an xyz command and served over three
// frontends — CLI subcommand, HTTP REST route and MCP tool — with one shared
// decode/validate/handler pipeline. It also owns the process-long engine
// cache so repeated searches (and the long-lived serve/mcp processes) keep
// the index in memory and auto-reload when gs watch swaps in a new one.
package xyzsvc

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ejfkdev/gs"
	"github.com/ejfkdev/xyz-go/registry"
	"github.com/ejfkdev/xyz-go/spec"
)

// liveReloadInterval 是引擎缓存的自动重载间隔; 与 gs watch 的原子替换配合,
// serve/mcp 长驻时能在索引更新后自动读到新数据。
const liveReloadInterval = 5 * time.Second

// Registry builds the gs command set: search / schema / index. Any definition
// error (bad command name, conflicting hints) surfaces here at startup.
func Registry() (*registry.Registry, error) {
	reg := registry.New()
	defs := []struct {
		name string
		reg  func(*registry.Registry) error
	}{
		{name: "search", reg: func(r *registry.Registry) error {
			_, err := spec.Define("search", searchHandler).
				Summary("搜索索引").
				Description("对指定索引目录执行混合检索（BM25 + BGE），返回按相关度排序的命中列表。").
				CLI(spec.CliHints{
					Fields: map[string]spec.CliFieldHint{
						"k": {Shorthand: "k"},
					},
					After: `示例:
  gs search --index ./indexes/wiki "nginx 配置" -k 5
  GS_INDEX=./indexes/wiki gs search "database backup" --json | jq .
  gs search --index ./indexes/wiki "192.168.1.1" --fields name --strict

提示:
  --fields 可重复, 只限定参与搜索的字段; 默认输出人读表格, --json 得到 JSON
  (HTTP/MCP 返回的也是这种 JSON: id/score/path/source/tags/fields/snippet)。`,
				}).
				HTTP(spec.HTTPHints{Method: "GET", Path: "/search"}).
				MCP(spec.MCPHints{Annotations: []string{"read", "idempotent"}}).
				Register(r)
			return err
		}},
		{name: "schema", reg: func(r *registry.Registry) error {
			_, err := spec.Define("schema", schemaHandler).
				Summary("查看索引 schema").
				Description("读取索引目录的 schema.json，返回库名与全部字段定义（类型/可搜索/可嵌入/权重等）。").
				CLI(spec.CliHints{
					After: `示例:
  gs schema ./indexes/wiki
  gs schema ./indexes/wiki --json

提示:
  索引目录用位置参数传入; 输出字段的 name/type/searchable/embeddable/weight 等。`,
				}).
				HTTP(spec.HTTPHints{Method: "GET", Path: "/schema"}).
				MCP(spec.MCPHints{Annotations: []string{"read", "idempotent"}}).
				Register(r)
			return err
		}},
		{name: "index", reg: func(r *registry.Registry) error {
			_, err := spec.Define("index", indexHandler).
				Summary("重建索引").
				Description("按配置 YAML（schema + sources + mapping）重建索引到输出目录。首次构建需提供 BGE 权重与词表，之后可从索引目录自动读取。").
				CLI(spec.CliHints{
					After: `示例:
  gs index --config index.yaml --output ./indexes/myindex \
      --bge_weights ./model/model.safetensors --bge_vocab ./model/vocab.txt

提示:
  首次构建需提供 bge_weights/bge_vocab; 之后从索引目录自动读取, 无需再传。
  emb_cache 可选, 用于增量重建 (只重算变更文档)。`,
				}).
				HTTP(spec.HTTPHints{Method: "POST", Path: "/index"}).
				MCP(spec.MCPHints{Annotations: []string{"write"}}).
				Register(r)
			return err
		}},
		{name: "fastsearch", reg: func(r *registry.Registry) error {
			_, err := spec.Define("fastsearch", fastSearchHandler).
				Summary("快速搜索（纯 BM25，文本作 query）").
				Description("把一整段文本当作查询做纯 BM25 检索，不跑 BGE，毫秒级返回。对应旧 CLI 的 --doc 能力。").
				CLI(spec.CliHints{
					Fields: map[string]spec.CliFieldHint{
						"k": {Shorthand: "k"},
					},
					After: `示例:
  gs fastsearch --index ./indexes/wiki "nginx 配置与部署的说明文档" -k 5
  GS_INDEX=./indexes/wiki gs fastsearch "docker compose 容器编排" --json

提示:
  纯 BM25 (不跑 BGE), 把整段文本当 query, 适合把长文档/文件内容直接丢进来。`,
				}).
				HTTP(spec.HTTPHints{Method: "GET", Path: "/fastsearch"}).
				MCP(spec.MCPHints{Annotations: []string{"read", "idempotent"}}).
				Register(r)
			return err
		}},
	}
	for _, d := range defs {
		if err := d.reg(reg); err != nil {
			return nil, fmt.Errorf("xyzsvc: register %q: %w", d.name, err)
		}
	}
	return reg, nil
}

// ---- search ----

// SearchArgs 是 search 命令的三通道入参: Query 在 CLI 是位置参数, 在
// HTTP/MCP 是 query 参数; 其余字段在三种通道下含义一致。
type SearchArgs struct {
	Index  string   `json:"index" desc:"索引目录" required:"true" cli:"env=GS_INDEX"`
	Query  string   `json:"query" desc:"查询关键词" required:"true" cli:"positional" http:"query"`
	K      int      `json:"k" desc:"返回条数" default:"10"`
	Fields []string `json:"fields" desc:"限定搜索字段（空=全部可搜索字段）"`
	Strict bool     `json:"strict" desc:"强制精确匹配增强（IP/域名/hash 等 token）"`
}

// hitOut 是命中结果的小写 JSON 输出结构 (三种通道共用, 与旧 CLI 一致)。
type hitOut struct {
	ID      string            `json:"id"`
	Score   float32           `json:"score"`
	Path    string            `json:"path"`
	Source  string            `json:"source"`
	Tags    []string          `json:"tags,omitempty"`
	Fields  map[string]string `json:"fields"`
	Snippet string            `json:"snippet"`
}

func searchHandler(ctx context.Context, in *SearchArgs) ([]hitOut, error) {
	if len(in.Fields) > 0 {
		if err := validateFields(in.Index, in.Fields); err != nil {
			return nil, err
		}
	}
	eng, err := engineFor(in.Index)
	if err != nil {
		return nil, err
	}
	hits, err := eng.Search(ctx, gs.SearchOptions{
		Query:  in.Query,
		Fields: in.Fields,
		TopK:   in.K,
		Strict: in.Strict,
	})
	if err != nil {
		return nil, err
	}
	out := make([]hitOut, 0, len(hits))
	for _, h := range hits {
		out = append(out, hitOut{
			ID:      h.ID,
			Score:   h.Score,
			Path:    h.Path,
			Source:  h.Source,
			Tags:    h.Tags,
			Fields:  h.Fields,
			Snippet: h.Snippet,
		})
	}
	return out, nil
}

// validateFields 用 schema.json 校验 --fields, 避免拼错字段名时被检索库
// 静默忽略 (resolveSearchFields 对未知字段做 continue, 会返回空结果)。
func validateFields(dir string, fields []string) error {
	schema, err := gs.LoadSchemaFromFile(filepath.Join(dir, "schema.json"))
	if err != nil {
		return fmt.Errorf("read schema: %w", err)
	}
	known := schema.FieldMap()
	for _, name := range fields {
		if _, ok := known[name]; !ok {
			return fmt.Errorf("unknown field %q (available: %s)", name, strings.Join(schema.FieldNames(), ", "))
		}
	}
	return nil
}

// ---- schema ----

type SchemaArgs struct {
	Index string `json:"index" desc:"索引目录" required:"true" cli:"positional" http:"query"`
}

type fieldOut struct {
	Name       string  `json:"name"`
	Type       string  `json:"type"`
	Searchable bool    `json:"searchable"`
	Embeddable bool    `json:"embeddable"`
	Weight     float32 `json:"weight"`
	Display    bool    `json:"display"`
	Snippet    bool    `json:"snippet"`
	Strict     bool    `json:"strict"`
}

type schemaOut struct {
	Name   string     `json:"name"`
	Fields []fieldOut `json:"fields"`
}

func schemaHandler(_ context.Context, in *SchemaArgs) (*schemaOut, error) {
	schema, err := gs.LoadSchemaFromFile(filepath.Join(in.Index, "schema.json"))
	if err != nil {
		return nil, fmt.Errorf("read schema: %w", err)
	}
	out := &schemaOut{Name: schema.Name, Fields: make([]fieldOut, 0, len(schema.Fields))}
	for _, f := range schema.Fields {
		out.Fields = append(out.Fields, fieldOut{
			Name:       f.Name,
			Type:       f.Type.String(),
			Searchable: f.Searchable,
			Embeddable: f.Embeddable,
			Weight:     f.FieldWeight,
			Display:    f.Display,
			Snippet:    f.Snippet,
			Strict:     f.Strict,
		})
	}
	return out, nil
}

// ---- index ----

type IndexArgs struct {
	Config        string `json:"config" desc:"索引配置 YAML 路径" required:"true"`
	Output        string `json:"output" desc:"索引输出目录" required:"true"`
	BGEWeights    string `json:"bge_weights" desc:"BGE 权重文件（HF model.safetensors），首次构建必填"`
	BGEVocab      string `json:"bge_vocab" desc:"BGE 词表文件（HF vocab.txt），首次构建必填"`
	MaxEmbedRunes int    `json:"max_embed_runes" desc:"每字段嵌入截断 rune 数（0=默认 512）"`
	EmbedCache    string `json:"emb_cache" desc:"持久化 embedding 缓存目录（增量重建）"`
}

type indexOut struct {
	Items   int `json:"items"`
	Skipped int `json:"skipped"`
}

func indexHandler(_ context.Context, in *IndexArgs) (*indexOut, error) {
	cfg, err := gs.LoadIndexConfig(in.Config)
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	if in.MaxEmbedRunes > 0 {
		cfg.Schema.MaxEmbedRunes = in.MaxEmbedRunes
	}
	var opts []gs.IndexBuildOption
	// 模型只需首次构建指定; 之后从输出目录的固定名 model.safetensors/vocab.txt 读取。
	if in.BGEWeights != "" || in.BGEVocab != "" {
		opts = append(opts, gs.IndexWithBGEPaths(in.BGEWeights, in.BGEVocab))
	}
	if in.EmbedCache != "" {
		opts = append(opts, gs.IndexWithEmbedCache(in.EmbedCache))
	}
	stats, err := cfg.Build(in.Output, opts...)
	if err != nil {
		return nil, err
	}
	return &indexOut{Items: stats.Items, Skipped: stats.Skipped}, nil
}

// ---- fastsearch ----

type FastSearchArgs struct {
	Index string `json:"index" desc:"索引目录" required:"true" cli:"env=GS_INDEX"`
	Text  string `json:"text" desc:"作为查询的文本" required:"true" cli:"positional" http:"query"`
	K     int    `json:"k" desc:"返回条数" default:"10"`
}

type fastHitOut struct {
	Idx     int     `json:"idx"`
	Path    string  `json:"path"`
	Source  string  `json:"source"`
	Name    string  `json:"name"`
	Desc    string  `json:"desc,omitempty"`
	Score   float32 `json:"score"`
	Snippet string  `json:"snippet"`
}

// fastSearchHandler 走 LiveEngine.FastSearch (纯 BM25, 不跑 BGE), 与 search
// 共用引擎缓存, 长驻 serve/mcp 时不反复读盘解析索引。
func fastSearchHandler(_ context.Context, in *FastSearchArgs) ([]fastHitOut, error) {
	if strings.TrimSpace(in.Text) == "" {
		return nil, fmt.Errorf("text is required")
	}
	eng, err := engineFor(in.Index)
	if err != nil {
		return nil, err
	}
	results := eng.FastSearch(in.Text, in.K)
	out := make([]fastHitOut, 0, len(results))
	for _, r := range results {
		out = append(out, fastHitOut{
			Idx:     r.Idx,
			Path:    r.Path,
			Source:  r.Source,
			Name:    r.Name,
			Desc:    r.Desc,
			Score:   r.Score,
			Snippet: r.Snippet,
		})
	}
	return out, nil
}

// ---- engine cache ----

var (
	engMu   sync.RWMutex
	engines = map[string]*gs.LiveEngine{}
)

// engineFor 返回某索引目录的长驻引擎 (自动重载)。首个调用者按目录打开并
// 缓存; 后续调用共享同一实例, 避免每次请求都重新解析 items.bin / 模型。
func engineFor(dir string) (*gs.LiveEngine, error) {
	engMu.RLock()
	e := engines[dir]
	engMu.RUnlock()
	if e != nil {
		return e, nil
	}

	engMu.Lock()
	defer engMu.Unlock()
	if e := engines[dir]; e != nil {
		return e, nil
	}
	info, err := os.Stat(dir)
	if err != nil {
		return nil, fmt.Errorf("index: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("index %q is not a directory", dir)
	}
	e, err = gs.OpenLive(dir, liveReloadInterval)
	if err != nil {
		return nil, fmt.Errorf("open index: %w", err)
	}
	engines[dir] = e
	return e, nil
}

// Close 关闭所有缓存的引擎 (并清空缓存)。下一次 engineFor 会从磁盘重新打开。
// 主要用于测试与嵌入场景的干净退出; CLI/serve/mcp 常驻进程由 OS 在退出时回收。
func Close() {
	engMu.Lock()
	for _, e := range engines {
		_ = e.Close()
	}
	engines = map[string]*gs.LiveEngine{}
	engMu.Unlock()
}
