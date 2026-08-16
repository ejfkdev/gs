// bge_batched.go - 批量版 BGE forward pass
//
// 优化点:
//   1. 所有 matvec 用 blas32.Gemv (单条) 或 blas32.Gemm (批量)
//   2. 支持 B=8 批量编码, 大幅摊销启动/调度开销
//   3. 对 padding token 用 -1e9 mask 处理
//
// 数据布局 (B=batch, T=max_seq_len, H=hidden, NH=heads, HD=head_dim, F=ffn):
//   hidden:    [B, T, H]    flat: b*T*H + t*H + h
//   Q, K, V:   [B, T, H]
//   scores:    [B, NH, T, T]  flat: b*NH*T*T + h*T*T + i*T + j
//   attnOut:   [B, T, H]
//   ffnMid:    [B, T, F]
//   ffnOut:    [B, T, H]
//
// 对于 Sgemm 调用:
//   - matvec: hidden @ W^T → reshape [B*T, H] @ [H, H] → [B*T, H]
//   - attn scores: Q @ K^T  → reshape [B*NH*T, HD] @ [B*NH*T, HD] → [B*NH*T, T]
//   - attn context: scores @ V → [B*NH*T, T] @ [B*NH*T, HD] → [B*NH*T, HD]

package gs

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

// EncodeBatch: 批量 BGE forward pass
// input:  [][]int B 个 token id 序列 (变长, 自动 pad 到 max(T_i))
// output: [][]float32 B 个 512 维 embedding (L2 归一化)
func (w *BGEWeights) EncodeBatch(tokenIDs [][]int) [][]float32 {
	if len(tokenIDs) == 0 {
		return nil
	}
	B := len(tokenIDs)

	// 找 max T
	Tmax := 0
	for _, ids := range tokenIDs {
		if len(ids) > Tmax {
			Tmax = len(ids)
		}
	}
	if Tmax > w.Cfg.MaxPos {
		Tmax = w.Cfg.MaxPos
	}
	if Tmax == 0 {
		Tmax = 1
	}

	// padding mask: 1 = real token, 0 = pad
	mask := make([]float32, B*Tmax)
	for b, ids := range tokenIDs {
		for t := 0; t < Tmax; t++ {
			if t < len(ids) {
				mask[b*Tmax+t] = 1.0
			} else {
				mask[b*Tmax+t] = 0.0
			}
		}
	}

	// 1. Embedding lookup
	cfg := w.Cfg
	H := cfg.Hidden
	hidden := w.embedBatch(tokenIDs, Tmax, B, H, mask)

	// 2. transformer layers
	for li := 0; li < cfg.Layers; li++ {
		hidden = w.transformerLayerBatch(hidden, Tmax, B, &w.Layers[li], mask)
	}

	// 3. [CLS] pooling + L2 normalize
	out := make([][]float32, B)
	for b := 0; b < B; b++ {
		cls := make([]float32, H)
		copy(cls, hidden[b*Tmax*H:b*Tmax*H+H])
		// L2 normalize
		var sq float32
		for _, x := range cls {
			sq += x * x
		}
		if sq > 0 {
			inv := 1.0 / float32(math.Sqrt(float64(sq)))
			for i := range cls {
				cls[i] *= inv
			}
		}
		out[b] = cls
	}
	return out
}

// embedBatch: 批量 embedding lookup + layer norm
// input:  tokenIDs B 个变长序列, Tmax = max len
// output: [B, Tmax, H] 浮点张量
func (w *BGEWeights) embedBatch(tokenIDs [][]int, Tmax, B, H int, mask []float32) []float32 {
	hidden := make([]float32, B*Tmax*H)
	// 对 padding 也分配 0 向量 (mask 后续会通过 layernorm 间接传播)
	for b, ids := range tokenIDs {
		// 复制 ids 到 padded, 多余位置填 padID
		padded := make([]int, Tmax)
		for t := 0; t < Tmax; t++ {
			if t < len(ids) {
				padded[t] = ids[t]
			} else {
				padded[t] = w.padIDForEmbed()
			}
		}
		for t, id := range padded {
			for h := 0; h < H; h++ {
				hidden[b*Tmax*H+t*H+h] = w.WordEmb[id*H+h] + w.PosEmb[t*H+h] + w.TypeEmb[0*H+h]
			}
		}
	}
	// 整 embedding 做 layer norm (per token)
	hiddenNorm := make([]float32, B*Tmax*H)
	for b := 0; b < B; b++ {
		for t := 0; t < Tmax; t++ {
			if mask[b*Tmax+t] == 0 {
				// padding: 直接置 0 (layernorm 输出也是 0, 因 gamma*0=0)
				continue
			}
			layerNorm(hidden[b*Tmax*H+t*H:b*Tmax*H+t*H+H],
				w.EmbLnGamma, w.EmbLnBeta,
				hiddenNorm[b*Tmax*H+t*H:b*Tmax*H+t*H+H],
				H, 1e-12)
		}
	}
	return hiddenNorm
}

// padIDForEmbed: 给 padding token 返回一个 0 向量用的 ID (实际不影响, 因为 mask=0)
// WordPieceTokenizer.padID 是真实 ID, 但这里我们用 0 (词表中通常 0 是 [PAD])
func (w *BGEWeights) padIDForEmbed() int {
	// 词表第 0 个 token 一般是 [PAD], BGE 也是
	return 0
}

// transformerLayerBatch: 批量 transformer layer
// input:  x [B, T, H], mask [B, T]
// output: [B, T, H]
func (w *BGEWeights) transformerLayerBatch(x []float32, T, B int, lw *BGELayerWeights, mask []float32) []float32 {
	cfg := w.Cfg
	H := cfg.Hidden
	NH := cfg.Heads
	HD := cfg.HeadDim
	BT := B * T

	// Q, K, V = x @ W^T + b  (用 Sgemm 批量)
	q := w.batchedMatMul(x, lw.QW, lw.QB, BT, H, H)
	k := w.batchedMatMul(x, lw.KW, lw.KB, BT, H, H)
	v := w.batchedMatMul(x, lw.VW, lw.VB, BT, H, H)

	// Attention: scores = Q @ K^T / sqrt(HD)
	// Q reshape: [B, NH, T, HD] (按 b, h, t, d 顺序)
	// K reshape: 同样
	// scores per (b, h): [T, T] = Q_block @ K_block^T
	BNHT := B * NH * T
	scale := float32(1.0 / math.Sqrt(float64(HD)))

	Qflat := make([]float32, BNHT*HD)
	Kflat := make([]float32, BNHT*HD)
	Vflat := make([]float32, BNHT*HD)
	// 重排: hidden[b, t, h*H + h_idx*HD + d] → Qflat[b*NH*T*HD + h_idx*T*HD + t*HD + d]
	for b := 0; b < B; b++ {
		for hi := 0; hi < NH; hi++ {
			for t := 0; t < T; t++ {
				srcOff := b*T*H + t*H + hi*HD
				dstOff := b*NH*T*HD + hi*T*HD + t*HD
				copy(Qflat[dstOff:dstOff+HD], q[srcOff:srcOff+HD])
				copy(Kflat[dstOff:dstOff+HD], k[srcOff:srcOff+HD])
				copy(Vflat[dstOff:dstOff+HD], v[srcOff:srcOff+HD])
			}
		}
	}

	// Per-(b,h) block: scores_block[T, T] = Q_block @ K_block^T
	// context_block[T, HD] = scores_block @ V_block
	contextFlat := make([]float32, BNHT*HD)
	for b := 0; b < B; b++ {
		for hi := 0; hi < NH; hi++ {
			blockOff := (b*NH + hi) * T * HD
			scores := make([]float32, T*T)

			// scores = scale * Q_block @ K_block^T
			Qgen := blas32.General{Rows: T, Cols: HD, Stride: HD, Data: Qflat[blockOff : blockOff+T*HD]}
			Kgen := blas32.General{Rows: T, Cols: HD, Stride: HD, Data: Kflat[blockOff : blockOff+T*HD]}
			Sgen := blas32.General{Rows: T, Cols: T, Stride: T, Data: scores}
			blas32.Gemm(blas.NoTrans, blas.Trans, scale, Qgen, Kgen, 0.0, Sgen)

			// Mask pad positions + softmax per row
			for i := 0; i < T; i++ {
				row := scores[i*T : (i+1)*T]
				for j := 0; j < T; j++ {
					if mask[b*T+j] == 0 {
						row[j] = -1e9
					}
				}
				// softmax
				var max float32 = row[0]
				for _, v := range row {
					if v > max {
						max = v
					}
				}
				var sum float32
				for j := 0; j < T; j++ {
					row[j] = float32(math.Exp(float64(row[j] - max)))
					sum += row[j]
				}
				if sum > 0 {
					inv := 1.0 / sum
					for j := 0; j < T; j++ {
						row[j] *= inv
					}
				}
			}

			// context_block = scores @ V_block
			Vgen := blas32.General{Rows: T, Cols: HD, Stride: HD, Data: Vflat[blockOff : blockOff+T*HD]}
			Cgen := blas32.General{Rows: T, Cols: HD, Stride: HD, Data: contextFlat[blockOff : blockOff+T*HD]}
			blas32.Gemm(blas.NoTrans, blas.NoTrans, 1.0, Sgen, Vgen, 0.0, Cgen)
		}
	}

	// 重排回 [B, T, H]: context[b, t, h] = contextFlat[b*NH*T*HD + h*T*HD + t*HD + d]
	context := make([]float32, B*T*H)
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			for hi := 0; hi < NH; hi++ {
				srcOff := b*NH*T*HD + hi*T*HD + t*HD
				dstOff := b*T*H + t*H + hi*HD
				copy(context[dstOff:dstOff+HD], contextFlat[srcOff:srcOff+HD])
			}
		}
	}

	// Output projection: proj = context @ W_O^T + b_O
	proj := w.batchedMatMul(context, lw.OW, lw.OB, BT, H, H)

	// Residual + LayerNorm
	attnNorm := make([]float32, B*T*H)
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			if mask[b*T+t] == 0 {
				continue
			}
			// x = x + proj
			sum := make([]float32, H)
			for h := 0; h < H; h++ {
				sum[h] = x[b*T*H+t*H+h] + proj[b*T*H+t*H+h]
			}
			layerNorm(sum, lw.AttnLnG, lw.AttnLnB, attnNorm[b*T*H+t*H:b*T*H+t*H+H], H, 1e-12)
		}
	}

	// FFN1: ffnMid = gelu(attnNorm @ W_FFN1^T + b_FFN1)
	F := cfg.FFN
	ffnMid := w.batchedMatMul(attnNorm, lw.FFN1W, lw.FFN1B, BT, F, H)
	// gelu
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			if mask[b*T+t] == 0 {
				continue
			}
			off := b*T*F + t*F
			for i := 0; i < F; i++ {
				ffnMid[off+i] = gelu(ffnMid[off+i])
			}
		}
	}
	// FFN2: ffnOut = ffnMid @ W_FFN2^T + b_FFN2
	ffnOut := w.batchedMatMul(ffnMid, lw.FFN2W, lw.FFN2B, BT, H, F)

	// Residual + LayerNorm
	out := make([]float32, B*T*H)
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			if mask[b*T+t] == 0 {
				continue
			}
			sum := make([]float32, H)
			for h := 0; h < H; h++ {
				sum[h] = attnNorm[b*T*H+t*H+h] + ffnOut[b*T*H+t*H+h]
			}
			layerNorm(sum, lw.FFNLnG, lw.FFNLnB, out[b*T*H+t*H:b*T*H+t*H+H], H, 1e-12)
		}
	}
	return out
}

// batchedMatMul: y = x @ W^T + b
//
//	x: [M, K] row-major
//	W: [N, K] row-major (即: 输出维度在前)
//	b: [N] (per-output bias) 或 nil
//	y: [M, N] row-major
//	M = BT, K = H, N = H (或 F)
func (w *BGEWeights) batchedMatMul(x, W, b []float32, M, N, K int) []float32 {
	y := make([]float32, M*N)
	Xgen := blas32.General{Rows: M, Cols: K, Stride: K, Data: x}
	Wgen := blas32.General{Rows: N, Cols: K, Stride: K, Data: W}
	Ygen := blas32.General{Rows: M, Cols: N, Stride: N, Data: y}
	// y = x @ W^T
	blas32.Gemm(blas.NoTrans, blas.Trans, 1.0, Xgen, Wgen, 0.0, Ygen)
	// add bias
	if b != nil {
		for m := 0; m < M; m++ {
			for n := 0; n < N; n++ {
				y[m*N+n] += b[n]
			}
		}
	}
	return y
}
