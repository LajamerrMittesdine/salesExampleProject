// Package ref is an idiomatic float32 educational port of Kimi K3 core algorithms.
// It mirrors kimi-k3/reference/python/kimi_k3_ref one-to-one against fixtures/goldens.json.
package ref

import "math"

func sigmoid(x float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-x))))
}

func tanh32(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

func exp32(x float32) float32 {
	return float32(math.Exp(float64(x)))
}

// RMSNorm matches KimiRMSNorm.
func RMSNorm(x, weight []float32, eps float32) []float32 {
	h := len(weight)
	out := make([]float32, len(x))
	rows := len(x) / h
	for r := 0; r < rows; r++ {
		base := r * h
		var v float32
		for j := 0; j < h; j++ {
			t := x[base+j]
			v += t * t
		}
		v /= float32(h)
		inv := 1 / float32(math.Sqrt(float64(v+eps)))
		for j := 0; j < h; j++ {
			out[base+j] = x[base+j] * inv * weight[j]
		}
	}
	return out
}

// SituAndMul matches SituAndMul; x last dim = lastDim (even).
func SituAndMul(x []float32, lastDim int, beta float32, linearBeta *float32) []float32 {
	d := lastDim / 2
	rows := len(x) / lastDim
	out := make([]float32, rows*d)
	for r := 0; r < rows; r++ {
		xs := x[r*lastDim : (r+1)*lastDim]
		for i := 0; i < d; i++ {
			gate := xs[i]
			up := xs[d+i]
			situA := beta * tanh32(gate/beta) * sigmoid(gate)
			if linearBeta != nil {
				lb := *linearBeta
				up = lb * tanh32(up/lb)
			}
			out[r*d+i] = situA * up
		}
	}
	return out
}

// Linear: y = x @ W.T, W shaped (outDim, inDim).
func Linear(x []float32, rows, inDim int, weight []float32, outDim int) []float32 {
	out := make([]float32, rows*outDim)
	for r := 0; r < rows; r++ {
		for o := 0; o < outDim; o++ {
			var s float32
			w := weight[o*inDim : (o+1)*inDim]
			xr := x[r*inDim : (r+1)*inDim]
			for j := 0; j < inDim; j++ {
				s += xr[j] * w[j]
			}
			out[r*outDim+o] = s
		}
	}
	return out
}

// MoEGate deterministic top-k.
func MoEGate(hidden []float32, n, h int, weight []float32, numExperts int, bias []float32, topK int, scale float32, renormalize bool) (idx []int64, wt []float32) {
	idx = make([]int64, n*topK)
	wt = make([]float32, n*topK)
	scores := make([]float32, numExperts)
	choice := make([]float32, numExperts)
	for row := 0; row < n; row++ {
		x := hidden[row*h : (row+1)*h]
		for e := 0; e < numExperts; e++ {
			var logit float32
			w := weight[e*h : (e+1)*h]
			for j := 0; j < h; j++ {
				logit += x[j] * w[j]
			}
			scores[e] = sigmoid(logit)
			choice[e] = scores[e] + bias[e]
		}
		order := make([]int, numExperts)
		for i := range order {
			order[i] = i
		}
		// sort by choice desc, tie smaller index
		for i := 0; i < numExperts; i++ {
			for j := i + 1; j < numExperts; j++ {
				swap := false
				if choice[order[j]] > choice[order[i]] {
					swap = true
				} else if choice[order[j]] == choice[order[i]] && order[j] < order[i] {
					swap = true
				}
				if swap {
					order[i], order[j] = order[j], order[i]
				}
			}
		}
		chosen := append([]int(nil), order[:topK]...)
		for i := 0; i < topK; i++ {
			for j := i + 1; j < topK; j++ {
				if chosen[j] < chosen[i] {
					chosen[i], chosen[j] = chosen[j], chosen[i]
				}
			}
		}
		for k := 0; k < topK; k++ {
			idx[row*topK+k] = int64(chosen[k])
			wt[row*topK+k] = scores[chosen[k]]
		}
		if topK > 1 && renormalize {
			var denom float32 = 1e-20
			for k := 0; k < topK; k++ {
				denom += wt[row*topK+k]
			}
			for k := 0; k < topK; k++ {
				wt[row*topK+k] /= denom
			}
		}
		for k := 0; k < topK; k++ {
			wt[row*topK+k] *= scale
		}
	}
	return idx, wt
}

// ExpertFFN SiTU expert.
func ExpertFFN(x []float32, rows, hidden int, w1, w2, w3 []float32, inter int, beta float32, linearBeta *float32) []float32 {
	gate := Linear(x, rows, hidden, w1, inter)
	up := Linear(x, rows, hidden, w3, inter)
	gateUp := make([]float32, rows*2*inter)
	for r := 0; r < rows; r++ {
		copy(gateUp[r*2*inter:r*2*inter+inter], gate[r*inter:(r+1)*inter])
		copy(gateUp[r*2*inter+inter:r*2*inter+2*inter], up[r*inter:(r+1)*inter])
	}
	act := SituAndMul(gateUp, 2*inter, beta, linearBeta)
	return Linear(act, rows, inter, w2, hidden)
}

// ApplyAttnRes matches _apply_attn_res.
func ApplyAttnRes(prefix []float32, n, hidden int, blockResidual []float32, numBlocks int, proj, norm []float32, eps float32) []float32 {
	blocks := numBlocks + 1
	v := make([]float32, n*blocks*hidden)
	for i := 0; i < n; i++ {
		for b := 0; b < numBlocks; b++ {
			copy(v[(i*blocks+b)*hidden:(i*blocks+b+1)*hidden],
				blockResidual[(i*numBlocks+b)*hidden:(i*numBlocks+b+1)*hidden])
		}
		copy(v[(i*blocks+numBlocks)*hidden:(i*blocks+numBlocks+1)*hidden],
			prefix[i*hidden:(i+1)*hidden])
	}
	scores := make([]float32, n*blocks)
	for i := 0; i < n; i++ {
		for b := 0; b < blocks; b++ {
			base := (i*blocks + b) * hidden
			var vr float32
			for j := 0; j < hidden; j++ {
				t := v[base+j]
				vr += t * t
			}
			vr /= float32(hidden)
			inv := 1 / float32(math.Sqrt(float64(vr+eps)))
			var score float32
			for j := 0; j < hidden; j++ {
				score += v[base+j] * inv * norm[j] * proj[j]
			}
			scores[i*blocks+b] = score
		}
	}
	probs := StableSoftmaxRows(scores, n, blocks)
	out := make([]float32, n*hidden)
	for i := 0; i < n; i++ {
		for j := 0; j < hidden; j++ {
			var acc float32
			for b := 0; b < blocks; b++ {
				acc += probs[i*blocks+b] * v[(i*blocks+b)*hidden+j]
			}
			out[i*hidden+j] = acc
		}
	}
	return out
}

// StableSoftmaxRows computes softmax over last axis.
func StableSoftmaxRows(x []float32, rows, cols int) []float32 {
	out := make([]float32, len(x))
	for r := 0; r < rows; r++ {
		row := x[r*cols : (r+1)*cols]
		m := float32(math.Inf(-1))
		for _, v := range row {
			if v > m {
				m = v
			}
		}
		var sum float32
		for c := 0; c < cols; c++ {
			e := exp32(row[c] - m)
			out[r*cols+c] = e
			sum += e
		}
		for c := 0; c < cols; c++ {
			out[r*cols+c] /= sum
		}
	}
	return out
}

// ShortConv1dSiLU causal depthwise conv + SiLU.
func ShortConv1dSiLU(x []float32, batch, time, channels int, weight []float32, kernel int) []float32 {
	out := make([]float32, len(x))
	for b := 0; b < batch; b++ {
		for t := 0; t < time; t++ {
			for c := 0; c < channels; c++ {
				var acc float32
				for i := 0; i < kernel; i++ {
					src := t + i - (kernel - 1)
					var xv float32
					if src >= 0 {
						xv = x[(b*time+src)*channels+c]
					}
					acc += xv * weight[c*kernel+i]
				}
				out[(b*time+t)*channels+c] = acc * sigmoid(acc) // SiLU(x) = x * sigmoid(x)
			}
		}
	}
	return out
}

// KDALowerBoundGate K3 gate.
func KDALowerBoundGate(g []float32, outer, heads, dim int, aLog []float32, dtBias []float32, lowerBound float32) []float32 {
	out := make([]float32, len(g))
	aExp := make([]float32, heads)
	for h := 0; h < heads; h++ {
		aExp[h] = exp32(aLog[h])
	}
	for o := 0; o < outer; o++ {
		for h := 0; h < heads; h++ {
			for d := 0; d < dim; d++ {
				idx := (o*heads+h)*dim + d
				v := g[idx]
				if dtBias != nil {
					v += dtBias[h*dim+d]
				}
				out[idx] = lowerBound * sigmoid(aExp[h]*v)
			}
		}
	}
	return out
}

// L2Normalize over last dim.
func L2Normalize(x []float32, rows, dim int, eps float32) []float32 {
	out := make([]float32, len(x))
	for r := 0; r < rows; r++ {
		var s float32
		for j := 0; j < dim; j++ {
			t := x[r*dim+j]
			s += t * t
		}
		inv := 1 / float32(math.Sqrt(float64(s+eps)))
		for j := 0; j < dim; j++ {
			out[r*dim+j] = x[r*dim+j] * inv
		}
	}
	return out
}

// KDARecurrent naive recurrent KDA.
func KDARecurrent(q, k, v, g, beta []float32, batch, time, heads, dim int, scale *float32, l2norm bool) (o, state []float32) {
	sc := float32(1 / math.Sqrt(float64(dim)))
	if scale != nil {
		sc = *scale
	}
	qq := append([]float32(nil), q...)
	kk := append([]float32(nil), k...)
	if l2norm {
		rows := batch * time * heads
		qq = L2Normalize(qq, rows, dim, 1e-6)
		kk = L2Normalize(kk, rows, dim, 1e-6)
	}
	for i := range qq {
		qq[i] *= sc
	}
	s := make([]float32, batch*heads*dim*dim)
	o = make([]float32, len(q))
	for t := 0; t < time; t++ {
		for b := 0; b < batch; b++ {
			for h := 0; h < heads; h++ {
				qBase := ((b*time+t)*heads + h) * dim
				sBase := (b*heads + h) * dim * dim
				bval := beta[(b*time+t)*heads+h]
				for kd := 0; kd < dim; kd++ {
					eg := exp32(g[qBase+kd])
					for vd := 0; vd < dim; vd++ {
						s[sBase+kd*dim+vd] *= eg
					}
				}
				ks := make([]float32, dim)
				for vd := 0; vd < dim; vd++ {
					var acc float32
					for kd := 0; kd < dim; kd++ {
						acc += kk[qBase+kd] * s[sBase+kd*dim+vd]
					}
					ks[vd] = acc
				}
				delta := make([]float32, dim)
				for vd := 0; vd < dim; vd++ {
					delta[vd] = v[qBase+vd] - ks[vd]
				}
				for kd := 0; kd < dim; kd++ {
					bk := bval * kk[qBase+kd]
					for vd := 0; vd < dim; vd++ {
						s[sBase+kd*dim+vd] += bk * delta[vd]
					}
				}
				for vd := 0; vd < dim; vd++ {
					var acc float32
					for kd := 0; kd < dim; kd++ {
						acc += qq[qBase+kd] * s[sBase+kd*dim+vd]
					}
					o[qBase+vd] = acc
				}
			}
		}
	}
	return o, s
}

// MLAEagerAttention causal eager attention; q/k/v (B,H,T,D) -> out (B,T,H,D).
func MLAEagerAttention(query, key, value []float32, batch, heads, time, dim int, scaling float32, causal bool) []float32 {
	scores := make([]float32, batch*heads*time*time)
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			for qi := 0; qi < time; qi++ {
				for kj := 0; kj < time; kj++ {
					var s float32
					qb := ((b*heads+h)*time + qi) * dim
					kb := ((b*heads+h)*time + kj) * dim
					for d := 0; d < dim; d++ {
						s += query[qb+d] * key[kb+d]
					}
					s *= scaling
					if causal && kj > qi {
						s = -1e9
					}
					scores[(((b*heads+h)*time)+qi)*time+kj] = s
				}
			}
		}
	}
	probs := StableSoftmaxRows(scores, batch*heads*time, time)
	tmp := make([]float32, len(query))
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			for qi := 0; qi < time; qi++ {
				for d := 0; d < dim; d++ {
					var acc float32
					for kj := 0; kj < time; kj++ {
						p := probs[(((b*heads+h)*time)+qi)*time+kj]
						vb := ((b*heads+h)*time + kj) * dim
						acc += p * value[vb+d]
					}
					tmp[((b*heads+h)*time+qi)*dim+d] = acc
				}
			}
		}
	}
	out := make([]float32, len(query))
	for b := 0; b < batch; b++ {
		for t := 0; t < time; t++ {
			for h := 0; h < heads; h++ {
				for d := 0; d < dim; d++ {
					out[((b*time+t)*heads+h)*dim+d] = tmp[((b*heads+h)*time+t)*dim+d]
				}
			}
		}
	}
	return out
}

// GatedMLAOutput applies sigmoid gate.
func GatedMLAOutput(attn, gateLogits []float32) []float32 {
	out := make([]float32, len(attn))
	for i := range attn {
		out[i] = attn[i] * sigmoid(gateLogits[i])
	}
	return out
}

// LatentMoEForward tiny LatentMoE path.
func LatentMoEForward(
	hidden []float32, batch, seq, hDim int,
	routerW, routerB []float32, numExperts, topK int,
	ew1, ew2, ew3 [][]float32,
	latent, inter int,
	routedDown, routedUp, routedNorm []float32,
	sharedGate, sharedUp, sharedDown []float32,
	rmsEps, beta float32, linearBeta *float32,
) []float32 {
	n := batch * seq
	idx, wt := MoEGate(hidden, n, hDim, routerW, numExperts, routerB, topK, 1, true)
	latentX := Linear(hidden, n, hDim, routedDown, latent)
	y := make([]float32, n*latent)
	for t := 0; t < n; t++ {
		xt := latentX[t*latent : (t+1)*latent]
		acc := make([]float32, latent)
		for k := 0; k < topK; k++ {
			e := int(idx[t*topK+k])
			tmp := ExpertFFN(xt, 1, latent, ew1[e], ew2[e], ew3[e], inter, beta, linearBeta)
			w := wt[t*topK+k]
			for j := 0; j < latent; j++ {
				acc[j] += w * tmp[j]
			}
		}
		copy(y[t*latent:(t+1)*latent], acc)
	}
	if routedNorm != nil {
		y = RMSNorm(y, routedNorm, rmsEps)
	}
	routed := Linear(y, n, latent, routedUp, hDim)
	shared := ExpertFFN(hidden, n, hDim, sharedGate, sharedDown, sharedUp, inter, beta, linearBeta)
	out := make([]float32, n*hDim)
	for i := range out {
		out[i] = routed[i] + shared[i]
	}
	return out
}

// DecoderBlockForward AttnRes residual algebra with scale stand-ins.
func DecoderBlockForward(
	hidden []float32, batch, seq, hDim int,
	blockResidual []float32, numBlocks int,
	inputNorm, postNorm, saProj, saNorm, mlpProj, mlpNorm []float32,
	layerIdx, blockSize int, attnScale, mlpScale, rmsEps float32,
) (prefixOut, brOut []float32, nbOut int) {
	n := batch * seq
	hiddenBuf := append([]float32(nil), hidden...)
	var prefix []float32 = append([]float32(nil), hidden...)
	br := append([]float32(nil), blockResidual...)
	nb := numBlocks

	if nb > 0 {
		hiddenBuf = ApplyAttnRes(prefix, n, hDim, br, nb, saProj, saNorm, rmsEps)
	}
	if layerIdx%blockSize == 0 {
		br = append(br, prefix...)
		nb++
		prefix = nil
	}
	normed := RMSNorm(hiddenBuf, inputNorm, rmsEps)
	for i := range normed {
		normed[i] *= attnScale
	}
	var prefixSum []float32
	if prefix != nil {
		prefixSum = make([]float32, n*hDim)
		for i := range prefixSum {
			prefixSum[i] = prefix[i] + normed[i]
		}
	} else {
		prefixSum = append([]float32(nil), normed...)
	}
	hidden2 := ApplyAttnRes(prefixSum, n, hDim, br, nb, mlpProj, mlpNorm, rmsEps)
	post := RMSNorm(hidden2, postNorm, rmsEps)
	for i := range post {
		post[i] *= mlpScale
	}
	for i := range prefixSum {
		prefixSum[i] += post[i]
	}
	return prefixSum, br, nb
}
