// safetensors.go - 读取 HuggingFace safetensors 模型权重
//
// safetensors 格式 = 8 字节 little-endian 的 JSON header 长度 + JSON header
// + 连续二进制 tensor 数据。JSON header 里每个 tensor 记录了 dtype / shape /
// data_offsets。纯 Go 读它非常简单, 所以 gs 直接读 HF 原始 model.safetensors,
// 它是唯一支持的权重格式 (pytorch_model.bin 是 pickle, 不解析; 也没有自定义
// 转换而来的中间格式)。
//
// 本文件按 bge-small-zh-v1.5 的 BertModel state_dict 键名映射到 BGEWeights。

package gs

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// defaultBGEHeads: bge-small-zh-v1.5 的注意力头数。safetensors 里无法从
// 权重形状反推出 heads (Q/K/V 都是 [H,H]), 而本 forward pass 只支持
// bge-small 架构, 故固定为 8。
const defaultBGEHeads = 8

type stTensorSpec struct {
	Dtype       string  `json:"dtype"`
	Shape       []int64 `json:"shape"`
	DataOffsets []int64 `json:"data_offsets"`
}

// parseSafetensors: 把 safetensors 字节解析为 BGEWeights
func parseSafetensors(data []byte) (*BGEWeights, error) {
	if len(data) < 8 {
		return nil, fmt.Errorf("safetensors: file too short")
	}
	headerLen := int(binary.LittleEndian.Uint64(data[:8]))
	if len(data) < 8+headerLen {
		return nil, fmt.Errorf("safetensors: truncated header")
	}

	var spec map[string]stTensorSpec
	if err := json.Unmarshal(data[8:8+headerLen], &spec); err != nil {
		return nil, fmt.Errorf("safetensors: bad header: %w", err)
	}
	body := data[8+headerLen:]

	// readF32: 按 spec 的 data_offsets 读出一个 float32 数组
	readF32 := func(name string, t stTensorSpec) ([]float32, error) {
		if t.Dtype != "F32" {
			return nil, fmt.Errorf("safetensors: tensor %q has unsupported dtype %q (only F32)", name, t.Dtype)
		}
		if len(t.DataOffsets) != 2 {
			return nil, fmt.Errorf("safetensors: tensor %q has bad data_offsets", name)
		}
		start, end := int(t.DataOffsets[0]), int(t.DataOffsets[1])
		if start < 0 || end > len(body) || start > end || (end-start)%4 != 0 {
			return nil, fmt.Errorf("safetensors: tensor %q has out-of-range offsets", name)
		}
		return bytesToFloat32LE(body[start:end]), nil
	}
	get := func(name string) ([]float32, error) {
		t, ok := spec[name]
		if !ok {
			return nil, fmt.Errorf("safetensors: missing tensor %q", name)
		}
		return readF32(name, t)
	}

	// 从 embedding 形状推导配置
	wordSpec, ok := spec["embeddings.word_embeddings.weight"]
	if !ok || len(wordSpec.Shape) != 2 {
		return nil, fmt.Errorf("safetensors: missing/invalid embeddings.word_embeddings.weight")
	}
	v, h := int(wordSpec.Shape[0]), int(wordSpec.Shape[1])

	posSpec := spec["embeddings.position_embeddings.weight"]
	p := 0
	if len(posSpec.Shape) == 2 {
		p = int(posSpec.Shape[0])
	}
	typeSpec := spec["embeddings.token_type_embeddings.weight"]
	tv := 0
	if len(typeSpec.Shape) == 2 {
		tv = int(typeSpec.Shape[0])
	}

	// 层数 = 出现过的最大 encoder.layer 编号 + 1; FFN = intermediate.dense 的第一维
	layers := 0
	ffn := 0
	for name, t := range spec {
		if strings.HasPrefix(name, "encoder.layer.") {
			rest := name[len("encoder.layer."):]
			dot := strings.IndexByte(rest, '.')
			if dot > 0 {
				if i, err := strconv.Atoi(rest[:dot]); err == nil && i+1 > layers {
					layers = i + 1
				}
			}
		}
		if name == "encoder.layer.0.intermediate.dense.weight" && len(t.Shape) == 2 {
			ffn = int(t.Shape[0])
		}
	}
	if layers == 0 {
		return nil, fmt.Errorf("safetensors: no encoder layers found")
	}
	if layers != 4 {
		return nil, fmt.Errorf("safetensors: unsupported layer count %d (this forward pass targets bge-small, 4 layers)", layers)
	}
	if ffn == 0 {
		return nil, fmt.Errorf("safetensors: missing encoder.layer.0.intermediate.dense.weight")
	}

	w := &BGEWeights{Cfg: BGEConfig{
		VocabSize: v, Hidden: h, Layers: layers, Heads: defaultBGEHeads,
		FFN: ffn, MaxPos: p, TypeVocab: tv, HeadDim: h / defaultBGEHeads,
	}}

	var err error
	// embeddings
	if w.WordEmb, err = get("embeddings.word_embeddings.weight"); err != nil {
		return nil, err
	}
	if w.PosEmb, err = get("embeddings.position_embeddings.weight"); err != nil {
		return nil, err
	}
	if w.TypeEmb, err = get("embeddings.token_type_embeddings.weight"); err != nil {
		return nil, err
	}
	if w.EmbLnGamma, err = get("embeddings.LayerNorm.weight"); err != nil {
		return nil, err
	}
	if w.EmbLnBeta, err = get("embeddings.LayerNorm.bias"); err != nil {
		return nil, err
	}

	// encoder layers
	w.Layers = make([]BGELayerWeights, layers)
	for i := 0; i < layers; i++ {
		lw := &w.Layers[i]
		pre := "encoder.layer." + strconv.Itoa(i) + "."
		if lw.QW, err = get(pre + "attention.self.query.weight"); err != nil {
			return nil, err
		}
		if lw.QB, err = get(pre + "attention.self.query.bias"); err != nil {
			return nil, err
		}
		if lw.KW, err = get(pre + "attention.self.key.weight"); err != nil {
			return nil, err
		}
		if lw.KB, err = get(pre + "attention.self.key.bias"); err != nil {
			return nil, err
		}
		if lw.VW, err = get(pre + "attention.self.value.weight"); err != nil {
			return nil, err
		}
		if lw.VB, err = get(pre + "attention.self.value.bias"); err != nil {
			return nil, err
		}
		if lw.OW, err = get(pre + "attention.output.dense.weight"); err != nil {
			return nil, err
		}
		if lw.OB, err = get(pre + "attention.output.dense.bias"); err != nil {
			return nil, err
		}
		if lw.AttnLnG, err = get(pre + "attention.output.LayerNorm.weight"); err != nil {
			return nil, err
		}
		if lw.AttnLnB, err = get(pre + "attention.output.LayerNorm.bias"); err != nil {
			return nil, err
		}
		if lw.FFN1W, err = get(pre + "intermediate.dense.weight"); err != nil {
			return nil, err
		}
		if lw.FFN1B, err = get(pre + "intermediate.dense.bias"); err != nil {
			return nil, err
		}
		if lw.FFN2W, err = get(pre + "output.dense.weight"); err != nil {
			return nil, err
		}
		if lw.FFN2B, err = get(pre + "output.dense.bias"); err != nil {
			return nil, err
		}
		if lw.FFNLnG, err = get(pre + "output.LayerNorm.weight"); err != nil {
			return nil, err
		}
		if lw.FFNLnB, err = get(pre + "output.LayerNorm.bias"); err != nil {
			return nil, err
		}
	}

	// pooler (可选: 推理走 CLS pooling, 不依赖 pooler)
	if t, ok := spec["pooler.dense.weight"]; ok {
		if w.PoolerW, err = readF32("pooler.dense.weight", t); err == nil {
			w.PoolerB, _ = get("pooler.dense.bias")
		}
	}
	return w, nil
}
