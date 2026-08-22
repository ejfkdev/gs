# gs

**Hybrid (BM25 + BGE) full-text search for local knowledge bases — pure Go, single binary, no cgo.**

`gs` 把本地的一堆 markdown / YAML 文档（wiki、笔记、Skills 语料、任意自定义 corpus）建成可搜索的 hybrid 索引，离线查询，毫秒级返回。项目同时提供 **库**（`import "github.com/ejfkdev/gs"`，包名 `gs`）和 **CLI**（单二进制 `gs`）。

```
$ gs search --index ./indexes/wiki "nginx 配置" -k 3
# 结果按相关度排序（加 --json 得到 JSON）

$ gs serve --addr :8080   # HTTP: GET /search /schema, POST /index, /openapi.json
$ gs mcp stdio            # MCP: search/schema/index 三个工具
```

## 特性

| | |
|---|---|
| **Hybrid 检索** | BM25 (sparse) + BGE-small-zh-v1.5 (dense) + 短语/精确/前缀/格式 strict 多信号融合 |
| **多字段 schema** | 字段名/类型/权重完全自定义，构建时写入索引目录的 `schema.json` |
| **schema 自动读取** | 检索时 `Load` 自动读回 `schema.json`，无需再指定字段定义；缺失才报错 |
| **Strict 模式** | 自动检测 IP / domain / email / hash / URL / path 等固定格式并按字段加权 |
| **Snippet 高亮** | 多 hit 聚类 + 强制不重叠 + `【token】` 标记 |
| **纯 Go** | 分词器为无词典 bigram 实现；BGE forward pass 用 blas32 sgemm 批量加速；**零 cgo**，`CGO_ENABLED=0` 亦可交叉编译 |
| **跨平台** | 数据库格式 little-endian + IEEE 754 + UTF-8，macOS / Linux 互拷即用 |
| **增量 build** | 已有 `emb_*.bin` 跳过，重 build 只算缺失的 field |
| **一次定义，三通道** | search/schema/index 用 [xyz-go](https://github.com/ejfkdev/xyz-go) 定义一次，同时作为 CLI 子命令、HTTP REST（含 OpenAPI）与 MCP 工具提供 |

## 安装

需要 Go 1.26+。推荐直接用 `go install` 安装 CLI（库 `github.com/ejfkdev/gs` 是其依赖，会自动解析）：

```bash
go install github.com/ejfkdev/gs/cmd/gs@latest
```

或从源码构建：

```bash
git clone https://github.com/ejfkdev/gs
cd gs
(cd cmd/gs && go build -o ../../bin/gs .)
```

跨平台交叉编译（纯 Go，无需 C 工具链）：

```bash
(cd cmd/gs && GOOS=linux  GOARCH=amd64 go build -o ../../bin/gs.linux-amd64  .)
(cd cmd/gs && GOOS=linux  GOARCH=arm64 go build -o ../../bin/gs.linux-arm64  .)
(cd cmd/gs && GOOS=darwin GOARCH=arm64 go build -o ../../bin/gs.darwin-arm64 .)
```

> **模块结构**：仓库是「库 + CLI」两个 Go module。根 `go.mod` 只含库依赖（`gonum` + `yaml.v3`），**不引入 xyz-go**；`cmd/gs/go.mod` 才 require xyz-go、fsnotify 等 CLI 依赖，并保持干净（无本地 replace，`go install` 才能解析）。本地与 CI 编译用根目录 `go.work` 的 `replace gs => .` 指向同仓库最新库源码（每次编译都是最新）；`go install` 不读 `go.work`，按 `cmd/gs/go.mod` 里 `require` 的版本号走 github/proxy 解析发布版本。所以 `import "github.com/ejfkdev/gs"` 当库用时，模块图里不会出现 xyz-go / MCP SDK。
>
> **发版**：打一个 `vX.Y.Z` tag 即触发 CI 交叉编译并发布二进制（库与 CLI 共用一个版本号）：
>
> ```bash
> ./scripts/release.sh v0.3.3   # 同时打 v0.3.3 与 cmd/gs/v0.3.3 并推送
> ```

## 快速上手

### 1. 准备 BGE 模型（可选）

`gs` 内嵌 BGE-small-zh-v1.5 的 forward pass；权重文件需自行下载，**不随仓库提交**（约 91MB）。没有模型时索引自动降级为纯 BM25，功能完整可用。

模型、下载地址、支持的权重格式见 **[MODEL.md](MODEL.md)**。简要版本：

- 模型：`BAAI/bge-small-zh-v1.5`
  - HuggingFace：https://huggingface.co/BAAI/bge-small-zh-v1.5
  - 镜像：https://hf-mirror.com/BAAI/bge-small-zh-v1.5
- 权重：`model.safetensors`（gs 原生读取，无需转换；`pytorch_model.bin` 是 pickle，不支持）
- 词表 `vocab.txt` 直链：
  - https://huggingface.co/BAAI/bge-small-zh-v1.5/resolve/main/vocab.txt
  - https://hf-mirror.com/BAAI/bge-small-zh-v1.5/resolve/main/vocab.txt

模型文件只需在**首次构建**时通过 `--bge-weights` / `--bge-vocab` 指定一次，构建时原样存进索引目录（固定名 `model.safetensors` / `vocab.txt`）；之后构建和搜索都从索引目录按固定名自动读取，无需再传。

### 2. Build 索引

```bash
# wiki 语料: ./wiki/**/*.md
gs build wiki \
  --source /path/to/wiki \
  --output ./indexes/wiki \
  --bge-weights ./model/model.safetensors \
  --bge-vocab   ./model/vocab.txt

# skills 语料: ./skills/<user>/<skill>/SKILL.md (Claude Skills 格式)
gs build skills \
  --source /path/to/skills \
  --output ./indexes/skills \
  --bge-weights ./model/model.safetensors \
  --bge-vocab   ./model/vocab.txt
```

首次构建会编码全部字段；后续重跑会增量跳过已存在的 `emb_*.bin`。BGE 语义向量默认取每个字段前 **512 个 rune**（模型 512 token 上限），可用 `--max-embed-runes <n>` 调整（英文长文可调大到 2048，tokenizer 会自己封顶）。

### 3. 搜索 / 查看 schema

`search`、`schema`、`index`、`fastsearch` 四个命令都是「一次定义、三通道调用」：既能当 CLI 子命令，也能通过 HTTP REST 和 MCP 工具调用（见下面的 `serve` / `mcp`）。

```bash
# 基础查询 (query 用位置参数; -k/--k 为 top-K)
gs search --index ./indexes/wiki "nginx 配置" -k 5

# JSON 输出 (默认是人读表格, --json 得到 JSON; HTTP/MCP 返回的就是这种 JSON)
gs search --index ./indexes/wiki "database backup" --json | jq .

# 限定搜索字段 (可重复: --fields name --fields description)
gs search --index ./indexes/wiki "192.168.1.1" --fields name --strict

# 快速搜索: 纯 BM25, 把整段文本当 query (对应旧 --doc; 文件内容用 $(cat f) 传入)
gs fastsearch --index ./indexes/wiki "nginx 配置与部署的说明文档" -k 5

# 查看 schema (字段定义)
gs schema ./indexes/wiki
gs schema ./indexes/wiki --json
```

> CLI 下可用环境变量 `GS_INDEX` 指定默认索引目录（`search` / `fastsearch` 用），省去每次 `--index`。

### 4. HTTP 服务（gs serve）

```bash
gs serve --addr :8080
```

| 路由 | 说明 |
|---|---|
| `GET  /search?index=DIR&query=...&k=10` | 搜索索引 |
| `GET  /fastsearch?index=DIR&text=...&k=10` | 快速 BM25 搜索（文本作 query） |
| `GET  /schema?index=DIR` | 查看索引 schema |
| `POST /index` | 重建索引（JSON body: `config`/`output`/`bge_weights`/`bge_vocab`/`max_embed_runes`/`emb_cache`） |
| `GET  /openapi.json` | OpenAPI 3 文档 |
| `GET  /healthz` | 健康检查 |
| `POST /mcp` | MCP streamable HTTP 工具端点（同一端口） |

```bash
curl 'localhost:8080/search?index=./indexes/wiki&query=nginx&k=3'
curl 'localhost:8080/schema?index=./indexes/wiki'
curl -s localhost:8080/openapi.json | jq .
```

`serve` 背后是 `gs.OpenLive` 自动重载：配合 `gs watch`，索引更新后无需重启服务。若 `serve` 要暴露到非本机，用 `gs serve --bearer tok1,tok2`（或 `--xyz.bearer`）给 REST 与 `/mcp` 加 Bearer 鉴权——`POST /index` 属写操作，建议加保护。

### 5. MCP 工具服务（gs mcp）

```bash
gs mcp stdio          # 本地 stdio (最常见)
gs mcp sse --addr :8080
gs mcp http --addr :8080
```

`search` / `fastsearch` / `schema` / `index` 自动成为 MCP 工具，输入/输出 schema 与 REST 同源。给 MCP 客户端（如 Claude、Cursor、Cherry Studio 等）配置 `gs mcp stdio` 即可直接检索本地知识库。

### 6. 跨平台传输索引

索引文件全部 little-endian，跨平台直接拷贝：

```bash
rsync -av --progress ./indexes/ user@linux:/data/gs/indexes/
```

## 配置驱动索引（--config）

不想写代码、又有固定字段结构的数据（JSON / YAML / XML 类的），可以用一个 YAML 配置声明「索引哪些文件、每个字段怎么取」：

```yaml
# index.yaml
schema:
  name: myindex
  name_field: title
  fields:
    - {name: title,   type: text,     searchable: true, embeddable: true, weight: 5.0, display: true}
    - {name: body,    type: longtext, searchable: true, embeddable: true, weight: 1.0, display: true, snippet: true}
    - {name: tags,    type: tags,     searchable: true, embeddable: true, weight: 3.0, display: true}

sources:
  - dir: ./data
    include: "**/*.json"        # glob 文件匹配
    format: json
    on_error: skip              # skip(默认) | fail
    mapping:
      title: "title"            # 点路径取值
      body:  "content"
      tags:  "keywords"         # 数组 → tags

  - dir: ./wiki
    include: "**/*.md"
    format: frontmatter         # yaml frontmatter + 正文 (skill 格式)
    mapping: { title: name, body: "__body__", tags: tags }

  - dir: ./notes
    include: "**/*.txt"
    format: text                # 整文件当一个字段
    mapping: { body: "__text__" }
```

```bash
gs build --config index.yaml --output ./indexes/myindex \
    --bge-weights ./model/model.safetensors --bge-vocab ./model/vocab.txt
```

| 项 | 说明 |
|---|---|
| 支持的格式 | `json` / `yaml` / `frontmatter`(skill) / `text` / `csv` / `ndjson`(jsonl) / `xlsx` |
| `include` | glob（支持 `**`，如 `**/*.json`、`docs/**/*.md`） |
| `mapping` | 字段 → 点路径（`title`、`a.b[0].c`）；frontmatter/text 用 `__body__`/`__text__` 取正文 |
| `on_error` | 解析失败策略：`skip`（默认，跳过并告警）或 `fail`（整体失败） |
| `type` | `text` / `longtext` / `tags` |

解析失败默认跳过（打日志不中断）；要严格失败就设 `on_error: fail`。`csv`/`xlsx`/`ndjson` 每行/每条一个文档（`csv`/`xlsx` 首行表头，`mapping` 填列名或 0 基列号；`ndjson` 每行一个 JSON 对象）。

配置驱动索引**也是公开库 API**，可以在自己的程序里直接构建：

```go
import "github.com/ejfkdev/gs"

cfg, _ := gs.LoadIndexConfig("index.yaml")
stats, err := cfg.Build("./indexes/myindex",
    gs.IndexWithBGEPaths("./model/model.safetensors", "./model/vocab.txt"),
    gs.IndexWithEmbedCache("./indexes/.embcache"), // 增量缓存: 次次只算变更文档
    gs.IndexWithProgress(func(stage string, cur, total int) { fmt.Printf("%s %d/%d\n", stage, cur, total) }),
)
// stats.Items / stats.Skipped
```

CLI 等价用 `gs build --config index.yaml --emb-cache ./indexes/.embcache ...`。增量缓存按「字段内容哈希」复用向量，新增/删除/改动单个文档只重算变化的部分（缓存会顺带清掉已删文档的孤儿条目）。

## 监控目录增量索引（gs watch）

`gs watch` 监听源目录，变化后自动重建，并用**原子替换**发布索引——搜索进程可以无锁并发读、不受更新影响：

```bash
gs watch --config index.yaml --output ./indexes/myindex \
    --bge-weights ./model/model.safetensors --bge-vocab ./model/vocab.txt \
    --interval 5s
```

- 每次变更做全量重建，建到临时目录后一次性 `rename` 换进 `--output`。
- **增量 embedding 缓存**：`gs watch` 自动在 `<output>.embcache` 维护按内容哈希的向量缓存，新增/修改文档只重算变化的部分，不是每次都从头编码。
- **变化检测**：`fsnotify` 即时触发为主，同时按 `--interval` 轮询扫源目录签名兜底（防丢事件）。
- 搜索进程 `Load`/`OpenLive` 时几乎总读到完整快照；自动重载对「目录短暂缺失」会做几次短重试兜底。
- 崩溃恢复：重启时清理残留的临时/备份目录并做一次全量重建，无需状态日志。
- **单 watcher 互斥**：锁文件 `<output>.watch.lock` 里记录 pid + 心跳；新 watcher 启动时若发现持有者 pid 已死（或心跳超时 30s）就抢占锁，避免 `kill` 后锁残留卡死。

## 搜索进程自动重载（库）

长驻的搜索进程用 `gs.OpenLive` 持有索引，它会按间隔检测索引目录变化并自动重载（重载瞬间 `Search` 短暂串行化）：

```go
eng, err := gs.OpenLive("./indexes/myindex", 5*time.Second)
defer eng.Close()
hits, _ := eng.Search(ctx, gs.SearchOptions{Query: "hello", TopK: 10})
```

配合 `gs watch` 的原子目录替换，两边不会冲突。

## 作为 Go 库使用

```go
import "github.com/ejfkdev/gs"

// 1. 定义 schema
schema := &gs.Schema{
    Name: "mycorpus",
    Fields: []gs.Field{
        {Name: "title", Type: gs.FieldText, Searchable: true, Embeddable: true,
         FieldWeight: 5.0, Display: true, Strict: true},
        {Name: "body", Type: gs.FieldLongText, Searchable: true, Embeddable: true,
         FieldWeight: 1.0, Display: true, Snippet: true},
        {Name: "tags", Type: gs.FieldTags, Searchable: true, Embeddable: true,
         FieldWeight: 3.0, Display: true},
    },
}

// 2. Build
b, _ := gs.NewBuilder(schema, "./indexes/mycorpus",
    gs.WithBGEPaths("./model/model.safetensors", "./model/vocab.txt"),
    gs.WithWorkers(runtime.NumCPU()))
defer b.Close()
b.Add(gs.Item{
    Path:   "docs/intro.md",
    Source: "myapp",
    Fields: map[string]string{"title": "Hello", "body": "...", "tags": "intro demo"},
})
b.Build()

// 3. Search
eng, _ := gs.Load("./indexes/mycorpus")
defer eng.Close()
hits, _ := eng.Search(ctx, gs.SearchOptions{Query: "hello", TopK: 10})
for _, h := range hits {
    fmt.Printf("%s (%.3f) %s\n  %s\n", h.ID, h.Score, h.Path, h.Snippet)
}
```

**`Load` 只在第一次解析磁盘文件**（schema + items + embeddings + 重建倒排索引）；`Search` 全程走内存、不碰磁盘。所以要 `Load` 一次、复用同一个 `*Engine` 反复 `Search`，而不是每次查询都重新 `Load`。

进程内需要「按目录缓存 + 同时持有多个不同索引实例」时用 `Registry`：

```go
reg := gs.NewRegistry()
wiki, _ := reg.Open("./indexes/wiki")      // 首次解析
wiki2, _ := reg.Open("./indexes/wiki")     // 复用同一个 *Engine (引用计数 +1)
skills, _ := reg.Open("./indexes/skills")   // 另一个独立实例
defer reg.CloseAll()
reg.Release(wiki); reg.Release(wiki2)       // 引用归零时自动关闭该实例
```

完整可运行示例：

- [examples/custom_indexer](examples/custom_indexer/) —— 用 `Builder` + `Add` 手工建索引（`go run ./examples/custom_indexer`）
- [examples/embedded_indexer](examples/embedded_indexer/) —— 声明式配置建索引 + `OpenLive` 自动重载 + 持续搜索（`go run ./examples/embedded_indexer`）

均无需外部文件即可跑通。

## Schema：随索引落盘、自动读取

**库不内置任何 schema**，字段定义由调用方自定义。构建时 schema 会连同索引一起写成 `schema.json` 存进索引目录；`Load` 会从索引目录自动读回这个 schema，检索时无需再传字段定义（`SearchOptions.Fields` 留空即搜全部可搜索字段）。索引目录里没有 `schema.json` 时 `Load` 才报错。

完全自定义 schema：

```go
schema := &gs.Schema{
    Name:      "products",
    NameField: "sku",        // 精确/前缀匹配与"名称"使用的字段; 留空自动推断
    MaxEmbedRunes: 0,        // BGE 每字段最大 rune 数; 0 = 默认 512
    Fields: []gs.Field{
        {Name: "sku",     Type: gs.FieldText,     Searchable: true, FieldWeight: 5.0, Display: true},
        {Name: "summary", Type: gs.FieldLongText, Searchable: true, FieldWeight: 1.0, Display: true, Snippet: true},
        {Name: "labels",  Type: gs.FieldTags,     Searchable: true, FieldWeight: 3.0, Display: true},
    },
}

// Item.Tags 会自动合并进所有 Type == FieldTags 的字段 (这里就是 labels),
// 因此字段可以叫 tags / labels / keywords 任意名字。
b.Add(gs.Item{
    Path:   "sku/123.md",
    Tags:   []string{"sale", "premium"},
    Fields: map[string]string{"sku": "widget-123", "summary": "..."},
})
```

字段语义完全由 Field 的旗标决定（`Searchable`/`Embeddable`/`Display`/`Snippet`/`Strict`/`FieldWeight`），与字段叫什么名无关；`NameField` 指定哪个字段承担精确/前缀匹配，不指定则自动取 `name` 字段或第一个短文本字段；`MaxEmbedRunes` 控制语义向量截断长度（0 = 默认 512）。校验用 `Schema.Validate()` / `NewSchema`。

> CLI 的 `gs build <skills|wiki>` 是比库更高层的便捷封装：它自带 skills/wiki 两种语料的 extractor 和字段定义（定义存在于 `internal/cli`，不是库 API），构建后统一落到 `schema.json`。

## 目录结构

```
gs/                             # module github.com/ejfkdev/gs (库)
├── *.go                        # gs 库本身 (import "github.com/ejfkdev/gs")
│   ├── types.go / schema.go    # 数据类型 + Schema
│   ├── engine.go               # BM25 + BGE + phrase + strict + rerank 主流程
│   ├── tokenize.go             # 纯 Go bigram 分词器 (无 cgo, 线程安全)
│   ├── bge.go / bge_batched.go # BGE forward (单条 + blas32 批量)
│   ├── snippet.go              # 多 hit snippet 提取
│   ├── strict.go               # IP/domain/hash 等固定格式检测
│   ├── build.go                # Builder (string interning + 并行编码)
│   └── load.go                 # Load (读 items.bin / emb_*.bin)
├── examples/                   # 可运行示例 (custom_indexer / embedded_indexer)
├── cmd/gs/                     # module github.com/ejfkdev/gs/cmd/gs (CLI)
│   ├── go.mod                  # require gs+xyz-go+fsnotify+yaml; replace gs => ../..
│   ├── main.go                 # CLI 入口 (单二进制)
│   └── internal/
│       ├── cli/                # build/watch/version 本地子命令 + root 派发
│       └── xyzsvc/             # xyz-go 命令定义 (search/fastsearch/schema/index) + 引擎缓存
└── bin/gs                      # 编译产物 (gitignore)
```

详见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 数据格式

`items.bin` + `emb_*.bin` + `model.safetensors` + `vocab.txt` + `schema.json`：

```
indexes/mycorpus/
├── items.bin            # 所有 item 的字段字符串 + 偏移表 (header + per-item offsets + buffers)
├── emb_<field>.bin      # 每个 Embeddable 字段的 float32 向量 (8B header N+dim + N*dim*4B)
├── model.safetensors    # BGE 模型权重 (HuggingFace safetensors 原样)
├── vocab.txt            # WordPiece vocab
└── schema.json          # schema 定义
```

## License

MIT