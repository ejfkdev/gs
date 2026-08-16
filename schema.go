// schema.go - Schema 定义和查询
// Schema 是字段集合, 描述一个索引库的结构

package gs

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// Schema: 字段集合
//
// Schema 完全由调用方定义: Fields 是任意名称、任意类型、任意权重的
// 字段列表, 库本身不内置任何 schema。构建时 Schema 会连同索引一起写入
// 磁盘 (schema.json), Load 时自动读回, 因此检索时无需再提供字段定义。
//
// NameField 指定用于精确/前缀匹配和"名称"展示的字段名; 留空时自动推断
// (优先 "name", 否则取第一个 Text+Display 字段)。tags 无需配置: 任何
// Type == FieldTags 的字段都会自动吸收 Item.Tags。
//
// MaxEmbedRunes 指定 BGE 编码时每个字段截断的最大 rune 数, 0 = 用默认
// DefaultMaxEmbedRunes (512)。该值会写进 schema.json, 构建和检索自动保持一致。
type Schema struct {
	Name          string // 库名
	Fields        []Field
	NameField     string // 精确/前缀匹配字段名; 空 = 自动推断
	MaxEmbedRunes int    // BGE 编码的字段最大 rune 数; 0 = DefaultMaxEmbedRunes
}

// PrimaryField: 返回用于精确/前缀匹配的字段名 (NameField 为空时自动推断)
func (s *Schema) PrimaryField() string {
	if s.NameField != "" {
		for _, f := range s.Fields {
			if f.Name == s.NameField {
				return f.Name
			}
		}
	}
	for _, f := range s.Fields {
		if f.Name == "name" {
			return f.Name
		}
	}
	for _, f := range s.Fields {
		if f.Display && f.Type == FieldText {
			return f.Name
		}
	}
	return ""
}

// primaryFieldIndex: PrimaryField 对应的字段下标 (-1 表示没有)
func (s *Schema) primaryFieldIndex() int {
	name := s.PrimaryField()
	for i, f := range s.Fields {
		if f.Name == name {
			return i
		}
	}
	return -1
}

// NewSchema: 创建一个 schema
func NewSchema(name string, fields []Field) (*Schema, error) {
	// 校验
	seen := make(map[string]bool)
	for i, f := range fields {
		if f.Name == "" {
			return nil, fmt.Errorf("field[%d]: name is empty", i)
		}
		if seen[f.Name] {
			return nil, fmt.Errorf("duplicate field name: %q", f.Name)
		}
		seen[f.Name] = true
		if f.FieldWeight == 0 {
			fields[i].FieldWeight = 1.0
		}
	}
	return &Schema{Name: name, Fields: fields}, nil
}

// Field: 获取字段 (按 name)
func (s *Schema) Field(name string) *Field {
	for i := range s.Fields {
		if s.Fields[i].Name == name {
			return &s.Fields[i]
		}
	}
	return nil
}

// SearchableFields: 返回参与 BM25 搜索的字段
func (s *Schema) SearchableFields() []Field {
	var out []Field
	for _, f := range s.Fields {
		if f.Searchable {
			out = append(out, f)
		}
	}
	return out
}

// EmbeddableFields: 返回参与 BGE embedding 的字段
func (s *Schema) EmbeddableFields() []Field {
	var out []Field
	for _, f := range s.Fields {
		if f.Embeddable {
			out = append(out, f)
		}
	}
	return out
}

// SnippetFields: 返回 snippet 提取的字段
func (s *Schema) SnippetFields() []Field {
	var out []Field
	for _, f := range s.Fields {
		if f.Snippet {
			out = append(out, f)
		}
	}
	return out
}

// DisplayFields: 返回搜索结果中显示的字段
func (s *Schema) DisplayFields() []Field {
	var out []Field
	for _, f := range s.Fields {
		if f.Display {
			out = append(out, f)
		}
	}
	return out
}

// StrictFields: 参与 strict mode 加权的字段
func (s *Schema) StrictFields() []Field {
	var out []Field
	for _, f := range s.Fields {
		if f.Strict {
			out = append(out, f)
		}
	}
	return out
}

// Validate: 校验 schema
func (s *Schema) Validate() error {
	if s.Name == "" {
		return fmt.Errorf("schema name is empty")
	}
	hasSearchable := false
	hasEmbeddable := false
	for i, f := range s.Fields {
		if f.Name == "" {
			return fmt.Errorf("field[%d]: name is empty", i)
		}
		if f.FieldWeight < 0 {
			return fmt.Errorf("field[%q]: weight must be >= 0", f.Name)
		}
		if f.Searchable {
			hasSearchable = true
		}
		if f.Embeddable {
			hasEmbeddable = true
		}
	}
	if !hasSearchable {
		return fmt.Errorf("schema must have at least one searchable field")
	}
	// NameField 若显式指定, 必须在字段列表中存在
	if s.NameField != "" && s.Field(s.NameField) == nil {
		return fmt.Errorf("NameField %q does not match any field", s.NameField)
	}
	if s.MaxEmbedRunes < 0 {
		return fmt.Errorf("MaxEmbedRunes must be >= 0")
	}
	// embeddable 是可选的: 没有则跳过 BGE, 纯 BM25 库
	_ = hasEmbeddable
	return nil
}

// ToJSON: 转 JSON (写到 schema.json)
func (s *Schema) ToJSON() ([]byte, error) {
	return json.MarshalIndent(s, "", "  ")
}

// FromJSON: 从 JSON 读
func SchemaFromJSON(data []byte) (*Schema, error) {
	var s Schema
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, err
	}
	return &s, nil
}

// LoadSchemaFromFile: 从 schema.json 读
func LoadSchemaFromFile(path string) (*Schema, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return SchemaFromJSON(data)
}

// SaveSchemaToFile: 写到 schema.json
func (s *Schema) SaveToFile(path string) error {
	data, err := s.ToJSON()
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// FieldNames: 返回所有字段名
func (s *Schema) FieldNames() []string {
	names := make([]string, len(s.Fields))
	for i, f := range s.Fields {
		names[i] = f.Name
	}
	return names
}

// FieldMap: 字段名 -> Field*
func (s *Schema) FieldMap() map[string]*Field {
	m := make(map[string]*Field, len(s.Fields))
	for i := range s.Fields {
		m[s.Fields[i].Name] = &s.Fields[i]
	}
	return m
}

// String: 友好打印
func (s *Schema) String() string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("Schema(%q):\n", s.Name))
	for _, f := range s.Fields {
		b.WriteString(fmt.Sprintf("  %-20s %-10s srch=%v emb=%v disp=%v snip=%v strict=%v w=%.1f\n",
			f.Name, f.Type.String(), f.Searchable, f.Embeddable, f.Display, f.Snippet, f.Strict, f.FieldWeight))
	}
	return b.String()
}
