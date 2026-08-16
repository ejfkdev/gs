// types.go - 公共数据类型定义
// Field, FieldType, Item, Hit, SearchOptions, SchemaInfo

package gs

// FieldType: 字段类型
type FieldType int

const (
	FieldText     FieldType = 0 // 短文本 (name, category, tags)
	FieldLongText FieldType = 1 // 长文本 (description, content) - snippet 来源
	FieldTags     FieldType = 2 // 标签数组
)

// String: 友好显示
func (t FieldType) String() string {
	switch t {
	case FieldText:
		return "text"
	case FieldLongText:
		return "longtext"
	case FieldTags:
		return "tags"
	default:
		return "unknown"
	}
}

// Field: 索引字段定义
// 注意: JSON 字段名用 Go 默认导出名 (Name/Type/...), 与 Schema 序列化风格一致.
type Field struct {
	Name        string    // 字段名
	Type        FieldType // 字段类型
	Searchable  bool      // 是否参与 BM25 搜索
	Embeddable  bool      // 是否参与 BGE embedding
	FieldWeight float32   // BM25 tf 加权
	Display     bool      // 搜索结果是否返回完整值
	Snippet     bool      // snippet 是否从该字段提取
	Strict      bool      // strict 模式加权字段
}

// Item: 原始文档 (用户输入)
type Item struct {
	ID     string            // 唯一标识 (可选; items.bin 不单独存 ID, 加载后 ID = Path)
	Path   string            // 原文档路径
	Source string            // 来源标记
	Tags   []string          // 标签; 构建时自动合并进每个 FieldTags 字段 (勿与 Fields 里的同名内容重复)
	Fields map[string]string // 字段名 -> 文本值
}

// Hit: 搜索结果
type Hit struct {
	ID      string            // 标识
	Idx     int               // 内部 item 索引
	Score   float32           // 最终综合分
	Path    string            // 原文档路径
	Source  string            // 来源标记
	Tags    []string          // 标签
	Fields  map[string]string // 各字段值 (按 Display)
	Snippet string            // 命中片段
	FTS5    float32           // BM25 分
	Emb     float32           // BGE 整体 cos 分
	Exact   bool              // name 精确匹配
	Prefix  bool              // 前缀匹配
	Strict  []string          // 命中的 strict token 类型
}

// SearchOptions: 搜索参数
type SearchOptions struct {
	Query  string   // 查询字符串
	Fields []string // 限定搜索字段 (空 = schema.SearchableFields())
	TopK   int      // 返回 top-K (默认 10)
	Strict bool     // 强制 strict 模式 (默认 auto: query 含 strict token 才启用)
	// 内部权重 (可选, 高级用户; 0 = 用默认值)
	WBM25     float32 // 默认 0.40
	WEmb      float32 // 默认 0.45
	WPhrase   float32 // 默认 0.10
	WExact    float32 // 默认 0.20
	WPrefix   float32 // 默认 0.10
	WStrict   float32 // 默认 0.30
	RerankTop int     // rerank 候选集大小 (默认 100)
	RerankW   float32 // rerank 权重 (默认 0.20)
}

// SchemaInfo: 索引库元信息 (写到 schema.json 顶层, 包含 schema 字段定义)
type SchemaInfo struct {
	Schema     *Schema `json:"-"`          // 关联的 schema
	Name       string  `json:"name"`       // 库名
	ItemCount  int     `json:"item_count"` // item 数
	EmbDim     int     `json:"emb_dim"`    // BGE 输出维度
	CreatedAt  string  `json:"created_at"` // 创建时间 (RFC3339)
	BGEVersion string  `json:"bge_version"`
	FormatVer  int     `json:"format_ver"` // 数据格式版本
}
