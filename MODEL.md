# 模型文件

gs 用 **BGE-small-zh-v1.5** 生成语义 embedding。模型权重约 91MB，**不提交进 git**，按下面方式获取并喂给 `gs build`。

## 模型信息

- 名称：`BAAI/bge-small-zh-v1.5`
- 架构：4 层 BERT，512 hidden，8 heads，FFN 2048，vocab 21128，max_pos 512
- 池化：CLS token + L2 归一化（推理侧已实现，权重按原始 BertModel 读取即可）

## 下载地址

- HuggingFace 官方：https://huggingface.co/BAAI/bge-small-zh-v1.5
- 国内镜像：https://hf-mirror.com/BAAI/bge-small-zh-v1.5

### 权重直链

- `model.safetensors`（唯一支持的权重格式）
  - https://huggingface.co/BAAI/bge-small-zh-v1.5/resolve/main/model.safetensors
  - https://hf-mirror.com/BAAI/bge-small-zh-v1.5/resolve/main/model.safetensors

### 词表直链

- `vocab.txt`
  - https://huggingface.co/BAAI/bge-small-zh-v1.5/resolve/main/vocab.txt
  - https://hf-mirror.com/BAAI/bge-small-zh-v1.5/resolve/main/vocab.txt

## 需要哪些文件

| 文件 | 说明 |
|---|---|
| `model.safetensors` | **唯一支持的权重格式**。safetensors = JSON header + 二进制 float32，gs 原生读取，无需任何转换 |
| `vocab.txt` | WordPiece 词表，直接当作 gs 的 `vocab.txt` 用 |
| `pytorch_model.bin` | Python pickle 序列化，**不支持**（pickle reader 又难又脆，不做）。需要的话用 HF 工具转成 safetensors |

## 用法

```bash
gs build wiki \
  --source /path/to/wiki \
  --output ./indexes/wiki \
  --bge-weights /path/to/BAAI/bge-small-zh-v1.5/model.safetensors \
  --bge-vocab   /path/to/BAAI/bge-small-zh-v1.5/vocab.txt
```

构建时 gs 把 `model.safetensors` 与 `vocab.txt` **原样存进索引目录**（固定名 `model.safetensors` / `vocab.txt`，权重不做任何转换）；之后的构建和搜索都从索引目录按固定名自动读取，`--bge-weights` / `--bge-vocab` 只需首次构建时传一次。

库用法等价：

```go
b, _ := gs.NewBuilder(schema, "./indexes/mycorpus",
    gs.WithBGEPaths("/path/to/BAAI/bge-small-zh-v1.5/model.safetensors",
                    "/path/to/BAAI/bge-small-zh-v1.5/vocab.txt"))
```