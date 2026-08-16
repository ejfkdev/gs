// indexcfg.go - 配置驱动的索引任务
//
// 一个 YAML 配置文件同时描述 schema（字段）与 sources（要索引哪些文件、
// 什么格式、每个字段怎么取值），替代 CLI 里写死的 skills/wiki 两条提取器。
// 用法: gs build --config index.yaml --output ./indexes/myindex

package cli

import (
	"fmt"
	"os"

	"github.com/ejfkdev/gs"
	"gopkg.in/yaml.v3"
)

// Config: 一个索引任务
type Config struct {
	Schema  SchemaConfig   `yaml:"schema"`
	Sources []SourceConfig `yaml:"sources"`
}

// SchemaConfig: schema 定义（字段名/类型/权重等，与库的 gs.Schema 对应）
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
	Format  string            `yaml:"format"`   // json | yaml | frontmatter | text | csv | xlsx
	OnError string            `yaml:"on_error"` // skip | fail（默认 skip）
	Mapping map[string]string `yaml:"mapping"`  // 字段名 -> 取值路径/列名
}

// LoadConfig: 读 YAML 配置
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	return &cfg, nil
}

// Schema: 转成库的 gs.Schema（并调用 Validate）
func (c *SchemaConfig) Schema() (*gs.Schema, error) {
	if c.Name == "" {
		return nil, fmt.Errorf("schema.name is required")
	}
	if len(c.Fields) == 0 {
		return nil, fmt.Errorf("schema.fields is required")
	}
	s := &gs.Schema{
		Name:          c.Name,
		NameField:     c.NameField,
		MaxEmbedRunes: c.MaxEmbedRunes,
		Fields:        make([]gs.Field, 0, len(c.Fields)),
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

// Field: 单个字段配置 → gs.Field
func (f *FieldConfig) Field() (gs.Field, error) {
	if f.Name == "" {
		return gs.Field{}, fmt.Errorf("field name is empty")
	}
	ft := gs.FieldText
	switch f.Type {
	case "", "text":
		ft = gs.FieldText
	case "longtext":
		ft = gs.FieldLongText
	case "tags":
		ft = gs.FieldTags
	default:
		return gs.Field{}, fmt.Errorf("field %q: unknown type %q (want text|longtext|tags)", f.Name, f.Type)
	}
	w := f.Weight
	if w == 0 {
		w = 1.0
	}
	return gs.Field{
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
