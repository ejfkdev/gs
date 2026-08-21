# gs 架构

## 目标

一个**纯 Go 的 hybrid (BM25 + BGE) 搜索库 + CLI**，服务于本地文档/知识库检索：
- 多独立索引库（skills / wiki / 自定义 corpus）
- 多字段 schema（name / description / tags / full_content 可配置）
- 搜索时可指定字段
- 搜索结果含原文档路径 + snippet 高亮
- 全文索引（不截断）
- 纯 Go，零 cgo，任意平台交叉编译（包括 `CGO_ENABLED=0`）

## 目录结构

```
gs/                                     # module: github.com/ejfkdev/gs (库)
├── *.go                                # gs 库 (package gs, import "github.com/ejfkdev/gs")
│   ├── types.go                        # Field / Item / Hit / SearchOptions
│   ├── schema.go                       # Schema (无内置 schema, 随索引落盘为 schema.json)
│   ├── engine.go                       # BM25 + BGE + phrase + strict + rerank + LRU
│   ├── tokenize.go                     # 纯 Go bigram 分词器 (无词典, 线程安全)
│   ├── bge.go                          # BGE forward (单条) + WordPiece + cache
│   ├── bge_batched.go                  # 批量 BGE forward (blas32 sgemm)
│   ├── snippet.go                      # 多 hit snippet (cluster + 不重叠), 高亮
│   ├── strict.go                       # IP/domain/hash 等固定格式检测 + boost
│   ├── build.go                        # Builder: interning + 并行 BGE + 增量复用
│   ├── load.go                         # Load: items.bin / emb_*.bin + inverted index 重建
│   ├── indexcfg.go / indexer_extract.go # 配置驱动索引 (公开 API)
│   └── live.go / registry.go / embcache.go
├── examples/custom_indexer/            # 用库构建自定义 corpus 的示例
├── examples/embedded_indexer/          # 声明式配置 + OpenLive 自动重载示例
└── cmd/gs/                             # module: github.com/ejfkdev/gs/cmd/gs (CLI)
    ├── go.mod                          # require gs + xyz-go + fsnotify + yaml; replace gs => ../..
    ├── main.go                         # CLI 入口
    └── internal/
        ├── cli/                        # build/watch/version 本地子命令 + root 派发
        └── xyzsvc/                     # xyz-go 命令定义 (search/fastsearch/schema/index) + 引擎缓存
```

## 公共 API

```go
import "github.com/ejfkdev/gs"

// Schema: 任意字段集合 (字段名/类型/权重都由调用方定义)
type Schema struct {
    Name      string  // 库名
    Fields    []Field
    NameField string  // 精确/前缀匹配字段名; 空 = 自动推断 ("name" 优先, 否则第一个 Text+Display)
}

// Field: 一个索引字段 (语义由旗标决定, 与字段名无关)
type Field struct {
    Name        string    // 字段名 (任意)
    Type        FieldType // Text / LongText / Tags
    Searchable  bool      // 是否参与 BM25 搜索
    Embeddable  bool      // 是否参与 BGE embedding
    FieldWeight float32   // BM25 tf 加权 (1.0=默认)
    Display     bool      // 是否在结果中返回完整值
    Snippet     bool      // snippet 是否从这个字段提取
    Strict      bool      // strict 模式加权字段
}

// Item.Tags 在构建时自动合并进所有 Type == FieldTags 的字段 (与字段名无关)。

// Item / Hit / SearchOptions (见 types.go)
// Engine: Load(dataDir) → Search(ctx, opts) → []Hit
// Builder: NewBuilder(schema, dataDir, opts...) → Add(item) → Build()
// Registry: NewRegistry() → Open(dir) / Release / Remove / CloseAll  (按目录缓存 + 多实例)
```

> **一次解析, 内存查询**: 磁盘读取 (schema.json / items.bin / emb_*.bin / BGE 权重)
> 只发生在 `Load` 里, `Engine.Search` 全程走内存、不重复读文件。`Registry`
> 按规范化目录缓存已加载的 `*Engine` 并做引用计数, 同一目录重复 `Open`
> 复用实例, 不同目录实例彼此独立。

## 搜索主流程 (engine.go)

```
query 进来
  ├─ tokenizeDedup (纯 Go bigram: 拉丁词 + CJK 双字重叠)
  ├─ 1. BM25   per-field, FieldWeight 加权
  ├─ 2. BGE    per-field embedding cos, FieldWeight 加权 (无模型 → hash 降级)
  ├─ 3. Phrase 相邻 token 短语命中加分 (只在 BM25 命中文档上算)
  ├─ 4. Exact / Prefix (预计算的 name 小写缓存)
  ├─ 5. Strict IP/domain/hash/... 固定格式自动检测 + 字段加权
  ├─ Stage 1: 归一化后加权求和 → sort
  ├─ Stage 2: rerank top-100 (token overlap + per-field emb cos)
  └─ top-K → buildHit (Display 字段 + snippet) → LRU 缓存 (key 覆盖 query/TopK/fields/weights)
```

## 分词器 (tokenize.go)

- 无词典、无 cgo、线程安全。
- 拉丁字母/数字连续段 → 单个 token (小写)。
- CJK 连续段 → 相邻双字组合 (bigram)；孤立的单个汉字才单独成 token。
- 索引构建与查询共用同一实现，切分必然一致；索引落盘只存原始文本，
  inverted index 在 Load 时并行重建，换分词器不需要重建索引。

> 历史版本使用 gojieba (cgo 封装 cppjieba)：与 "纯 Go / 零 cgo" 定位矛盾、
> `CGO_ENABLED=0` 无法编译、`init()` 内全局初始化对库使用者有副作用
> (`debug.SetGCPercent(50)`)。已替换为纯 Go 实现。

## 数据格式 (.bin)

```
dataDir/
├── items.bin            # items + 所有 string buffer
├── emb_<field>.bin      # 每个 embeddable field 一个 (搜索按字段加权使用 per-field 向量)
├── model.safetensors    # BGE 模型 (HuggingFace safetensors 原样, Load 自动读取)
├── vocab.txt            # BGE 词表 (固定名, Load 自动读取)
└── schema.json          # Schema 定义 (人类可读)
```

**items.bin 格式**:
```
header:
  uint32 magic "SKIH"    # 0x48494B53
  uint32 N               # item 数
  uint32 field_count
  uint32 path_buf_size
  uint32 tags_buf_size
  uint32 field_buf_sizes[field_count]
  uint32 source_buf_size
per-item:
  uint32 field_offsets[field_count]
  uint32 path_offset
  uint32 tags_offset
  uint8  source_len
  bytes  source_str
buffers:
  field_buf_0 .. field_buf_{field_count-1}   # null-terminated 字符串
  paths_buf | tags_buf | sources_buf
```

**emb_*.bin 格式**: `uint32 N | uint32 dim | float32[N*dim]`

注意: items.bin 不单独存 Item.ID，加载后 ID = Path（build 阶段 extractor
的 ID 本来就是 relPath，语义不变）。所有字段 little-endian。