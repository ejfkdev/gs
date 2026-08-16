package gs

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"strconv"
	"testing"
)

type tens struct {
	name  string
	shape []int
	data  []float32
}

// encodeSafetensors: 用给定 tensors 构造一个合法的 safetensors 字节流
func encodeSafetensors(tensors []tens) []byte {
	header := make(map[string]map[string]any)
	off := 0
	for _, t := range tensors {
		nbytes := len(t.data) * 4
		header[t.name] = map[string]any{
			"dtype":        "F32",
			"shape":        t.shape,
			"data_offsets": []int{off, off + nbytes},
		}
		off += nbytes
	}
	headerJSON, _ := json.Marshal(header)
	buf := make([]byte, 8+len(headerJSON)+off)
	binary.LittleEndian.PutUint64(buf[:8], uint64(len(headerJSON)))
	copy(buf[8:], headerJSON)
	pos := 8 + len(headerJSON)
	for _, t := range tensors {
		for _, v := range t.data {
			binary.LittleEndian.PutUint32(buf[pos:], math.Float32bits(v))
			pos += 4
		}
	}
	return buf
}

func zeros(n int) []float32 { return make([]float32, n) }

func TestParseSafetensors(t *testing.T) {
	const V, H, P, TV, L, F = 32, 8, 4, 2, 4, 16

	var tensors []tens
	add := func(name string, shape []int, data []float32) {
		tensors = append(tensors, tens{name, shape, data})
	}

	wordEmb := make([]float32, V*H)
	for i := range wordEmb {
		wordEmb[i] = float32(100 + i)
	}
	add("embeddings.word_embeddings.weight", []int{V, H}, wordEmb)
	add("embeddings.position_embeddings.weight", []int{P, H}, zeros(P*H))
	add("embeddings.token_type_embeddings.weight", []int{TV, H}, zeros(TV*H))
	add("embeddings.LayerNorm.weight", []int{H}, zeros(H))
	add("embeddings.LayerNorm.bias", []int{H}, zeros(H))

	for i := 0; i < L; i++ {
		pre := "encoder.layer." + strconv.Itoa(i) + "."
		add(pre+"attention.self.query.weight", []int{H, H}, zeros(H*H))
		add(pre+"attention.self.query.bias", []int{H}, zeros(H))
		add(pre+"attention.self.key.weight", []int{H, H}, zeros(H*H))
		add(pre+"attention.self.key.bias", []int{H}, zeros(H))
		add(pre+"attention.self.value.weight", []int{H, H}, zeros(H*H))
		add(pre+"attention.self.value.bias", []int{H}, zeros(H))
		add(pre+"attention.output.dense.weight", []int{H, H}, zeros(H*H))
		add(pre+"attention.output.dense.bias", []int{H}, zeros(H))
		add(pre+"attention.output.LayerNorm.weight", []int{H}, zeros(H))
		add(pre+"attention.output.LayerNorm.bias", []int{H}, zeros(H))
		add(pre+"intermediate.dense.weight", []int{F, H}, zeros(F*H))
		add(pre+"intermediate.dense.bias", []int{F}, zeros(F))
		add(pre+"output.dense.weight", []int{H, F}, zeros(H*F))
		add(pre+"output.dense.bias", []int{H}, zeros(H))
		add(pre+"output.LayerNorm.weight", []int{H}, zeros(H))
		add(pre+"output.LayerNorm.bias", []int{H}, zeros(H))
	}
	add("pooler.dense.weight", []int{H, H}, zeros(H*H))
	add("pooler.dense.bias", []int{H}, zeros(H))

	w, err := parseSafetensors(encodeSafetensors(tensors))
	if err != nil {
		t.Fatalf("parseSafetensors: %v", err)
	}

	cfg := w.Cfg
	if cfg.VocabSize != V || cfg.Hidden != H || cfg.Layers != L || cfg.Heads != 8 ||
		cfg.FFN != F || cfg.MaxPos != P || cfg.TypeVocab != TV {
		t.Fatalf("bad config: %+v", cfg)
	}
	if len(w.WordEmb) != V*H || w.WordEmb[0] != 100 || w.WordEmb[V*H-1] != float32(100+V*H-1) {
		t.Fatalf("word embeddings not read correctly: len=%d first=%v last=%v", len(w.WordEmb), w.WordEmb[0], w.WordEmb[V*H-1])
	}
	if len(w.Layers) != L {
		t.Fatalf("layers = %d, want %d", len(w.Layers), L)
	}
	if len(w.Layers[0].QW) != H*H || len(w.Layers[0].FFN1W) != F*H {
		t.Fatalf("layer-0 weight sizes wrong: QW=%d FFN1W=%d", len(w.Layers[0].QW), len(w.Layers[0].FFN1W))
	}
	if len(w.PoolerW) != H*H {
		t.Fatalf("pooler len = %d, want %d", len(w.PoolerW), H*H)
	}
}

func TestEffectiveMaxRunes(t *testing.T) {
	if effectiveMaxRunes(0) != DefaultMaxEmbedRunes {
		t.Errorf("effectiveMaxRunes(0) = %d, want %d", effectiveMaxRunes(0), DefaultMaxEmbedRunes)
	}
	if effectiveMaxRunes(300) != 300 {
		t.Errorf("effectiveMaxRunes(300) = %d, want 300", effectiveMaxRunes(300))
	}
	if DefaultMaxEmbedRunes <= 100 {
		t.Errorf("DefaultMaxEmbedRunes = %d, should be > 100 (full 512-token context)", DefaultMaxEmbedRunes)
	}
}
