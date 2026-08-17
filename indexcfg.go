// indexcfg.go - 配置驱动的索引（公开库 API）
//
// 用一份 YAML 声明 schema（字段）+ sources（要索引哪些文件、什么格式、每个字段
// 怎么取值），无需写提取代码即可构建索引：
//
//	cfg, _ := gs.LoadIndexConfig("index.yaml")
//	stats, err := cfg.Build("./indexes/myindex", gs.IndexWithBGEPaths(model, vocab))

package gs

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// IndexConfig: 一个索引任务（schema + sources）
type IndexConfig struct {
	Schema  SchemaConfig   `yaml:"schema"`
	Sources []SourceConfig `yaml:"sources"`
}

// SchemaConfig: schema 定义（与库的 Schema 对应）
type SchemaConfig struct {
	Name          string        `yaml:"name"`
	NameField     string        `yaml:"name_field"`
	MaxEmbedRunes int           `yaml:"max_embed_runes"`
	Fields        []FieldConfig `yaml:"fields"`
}

// FieldConfig: 单个字段
type FieldConfig struct {
	Name       string  `yaml:"name"`
	Type       string  `yaml:"type"` // text | longtext | tags
	Searchable bool    `yaml:"searchable"`
	Embeddable bool    `yaml:"embeddable"`
	Weight     float32 `yaml:"weight"`
	Display    bool    `yaml:"display"`
	Snippet    bool    `yaml:"snippet"`
	Strict     bool    `yaml:"strict"`
}

// SourceConfig: 一个数据源（目录 + glob + 格式 + 字段映射）
type SourceConfig struct {
	Dir     string            `yaml:"dir"`
	Include string            `yaml:"include"`  // glob，如 "**/*.json"
	Format  string            `yaml:"format"`   // json | yaml | frontmatter | text | csv | ndjson|jsonl | xlsx
	OnError string            `yaml:"on_error"` // skip（默认）| fail
	Mapping map[string]string `yaml:"mapping"`  // 字段名 -> 取值路径/列名
}

// IndexBuildStats: 构建/扫描统计
type IndexBuildStats struct {
	Items   int // 成功提取的文档数
	Skipped int // 提取失败跳过的文件数
}

// LoadIndexConfig: 从 YAML 文件读配置
func LoadIndexConfig(path string) (*IndexConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	return ParseIndexConfig(data)
}

// ParseIndexConfig: 从 YAML 字节解析配置
func ParseIndexConfig(data []byte) (*IndexConfig, error) {
	var cfg IndexConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	return &cfg, nil
}

// Schema: 转成库的 *Schema（并做 Validate）
func (c *SchemaConfig) Schema() (*Schema, error) {
	if c.Name == "" {
		return nil, fmt.Errorf("schema.name is required")
	}
	if len(c.Fields) == 0 {
		return nil, fmt.Errorf("schema.fields is required")
	}
	s := &Schema{
		Name:          c.Name,
		NameField:     c.NameField,
		MaxEmbedRunes: c.MaxEmbedRunes,
		Fields:        make([]Field, 0, len(c.Fields)),
	}
	for i := range c.Fields {
		f, err := c.Fields[i].Field()
		if err != nil {
			return nil, err
		}
		s.Fields = append(s.Fields, f)
	}
	if err := s.Validate(); err != nil {
		return nil, err
	}
	return s, nil
}

// Field: 单个字段配置 → Field
func (f *FieldConfig) Field() (Field, error) {
	if f.Name == "" {
		return Field{}, fmt.Errorf("field name is empty")
	}
	ft := FieldText
	switch f.Type {
	case "", "text":
		ft = FieldText
	case "longtext":
		ft = FieldLongText
	case "tags":
		ft = FieldTags
	default:
		return Field{}, fmt.Errorf("field %q: unknown type %q (want text|longtext|tags)", f.Name, f.Type)
	}
	w := f.Weight
	if w == 0 {
		w = 1.0
	}
	return Field{
		Name:        f.Name,
		Type:        ft,
		Searchable:  f.Searchable,
		Embeddable:  f.Embeddable,
		FieldWeight: w,
		Display:     f.Display,
		Snippet:     f.Snippet,
		Strict:      f.Strict,
	}, nil
}

// ------------------------------------------------------------------ 构建选项

type indexBuildOptions struct {
	bgeWeights  string
	bgeVocab    string
	workers     int
	embCacheDir string
	progress    func(stage string, cur, total int)
	onError     func(path string, err error)
}

// IndexBuildOption: 配置驱动的构建选项
type IndexBuildOption func(*indexBuildOptions)

// IndexWithBGEPaths: 设置 BGE 权重/词表路径（用于 embedding）
func IndexWithBGEPaths(weightsPath, vocabPath string) IndexBuildOption {
	return func(o *indexBuildOptions) {
		o.bgeWeights = weightsPath
		o.bgeVocab = vocabPath
	}
}

// IndexWithWorkers: BGE 编码并行度
func IndexWithWorkers(n int) IndexBuildOption {
	return func(o *indexBuildOptions) { o.workers = n }
}

// IndexWithEmbedCache: 设置持久化 embedding 缓存目录 (增量重建, 只算变更文档)
func IndexWithEmbedCache(cacheDir string) IndexBuildOption {
	return func(o *indexBuildOptions) { o.embCacheDir = cacheDir }
}

// IndexWithProgress: 构建进度回调
func IndexWithProgress(fn func(stage string, cur, total int)) IndexBuildOption {
	return func(o *indexBuildOptions) { o.progress = fn }
}

// IndexWithOnError: 提取失败回调（skip 模式下每个失败文件调用一次）
func IndexWithOnError(fn func(path string, err error)) IndexBuildOption {
	return func(o *indexBuildOptions) { o.onError = fn }
}

// Build: 提取所有 source 并把索引建到 outputDir，返回统计
func (c *IndexConfig) Build(outputDir string, opts ...IndexBuildOption) (IndexBuildStats, error) {
	schema, err := c.Schema.Schema()
	if err != nil {
		return IndexBuildStats{}, err
	}
	ib := resolveIndexBuildOptions(opts)

	builderOpts := make([]BuilderOption, 0, 3)
	if ib.progress != nil {
		builderOpts = append(builderOpts, WithProgress(ib.progress))
	}
	if ib.bgeWeights != "" {
		builderOpts = append(builderOpts, WithBGEPaths(ib.bgeWeights, ib.bgeVocab))
	}
	if ib.workers > 0 {
		builderOpts = append(builderOpts, WithWorkers(ib.workers))
	}
	if ib.embCacheDir != "" {
		builderOpts = append(builderOpts, WithEmbedCache(ib.embCacheDir))
	}
	builder, err := NewBuilder(schema, outputDir, builderOpts...)
	if err != nil {
		return IndexBuildStats{}, err
	}
	defer builder.Close()

	stats, err := c.walk(ib.onError, func(it Item) error { return builder.Add(it) })
	if err != nil {
		return stats, err
	}
	if stats.Items == 0 {
		return stats, fmt.Errorf("no documents matched any source")
	}
	if err := builder.Build(); err != nil {
		return stats, err
	}
	return stats, nil
}

// Scan: 只提取统计、不落盘（等价于 dry-run）
func (c *IndexConfig) Scan(opts ...IndexBuildOption) (IndexBuildStats, error) {
	if _, err := c.Schema.Schema(); err != nil {
		return IndexBuildStats{}, err
	}
	ib := resolveIndexBuildOptions(opts)
	return c.walk(ib.onError, nil)
}

func resolveIndexBuildOptions(opts []IndexBuildOption) indexBuildOptions {
	ib := indexBuildOptions{}
	for _, o := range opts {
		o(&ib)
	}
	return ib
}

// walk: 遍历所有 source，对每个匹配文件提取成 Item 并回调 onItem。
// 提取失败按 source.OnError 决定 skip（默认）还是 fail。
func (c *IndexConfig) walk(onError func(path string, err error), onItem func(Item) error) (IndexBuildStats, error) {
	var stats IndexBuildStats
	tagsFields := map[string]bool{}
	if schema, err := c.Schema.Schema(); err == nil {
		for _, f := range schema.Fields {
			if f.Type == FieldTags {
				tagsFields[f.Name] = true
			}
		}
	}
	for si, src := range c.Sources {
		if src.Dir == "" {
			return stats, fmt.Errorf("sources[%d].dir is required", si)
		}
		if src.Format == "" {
			return stats, fmt.Errorf("sources[%d].format is required", si)
		}
		include := src.Include
		if include == "" {
			include = "**"
		}
		onErr := src.OnError
		if onErr == "" {
			onErr = "skip"
		}
		if onErr != "skip" && onErr != "fail" {
			return stats, fmt.Errorf("sources[%d].on_error must be 'skip' or 'fail'", si)
		}
		err := filepath.WalkDir(src.Dir, func(p string, d fs.DirEntry, werr error) error {
			if werr != nil {
				return nil // 跳过无法进入的子目录
			}
			if d.IsDir() {
				return nil
			}
			rel, rerr := filepath.Rel(src.Dir, p)
			if rerr != nil {
				rel = p
			}
			if !MatchGlob(include, filepath.ToSlash(rel)) {
				return nil
			}
			items, err := extractSource(src, tagsFields, p, rel)
			if err != nil {
				stats.Skipped++
				if onError != nil {
					onError(rel, err)
				}
				if onErr == "fail" {
					return fmt.Errorf("extract %s: %w", rel, err)
				}
				return nil
			}
			for j := range items {
				if items[j].ID == "" {
					items[j].ID = rel
				}
				if items[j].Path == "" {
					items[j].Path = rel
				}
				items[j].Source = c.Schema.Name
				stats.Items++
				if onItem != nil {
					if err := onItem(items[j]); err != nil {
						return err
					}
				}
			}
			return nil
		})
		if err != nil {
			return stats, fmt.Errorf("walk %s: %w", src.Dir, err)
		}
	}
	return stats, nil
}
