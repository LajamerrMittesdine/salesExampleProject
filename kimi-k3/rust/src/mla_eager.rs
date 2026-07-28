use crate::attn_res::stable_softmax_rows;

/// Eager MLA attention. q/k/v: (b, h, t, d) flat. out: (b, t, h, d)
pub fn mla_eager_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    batch: usize,
    heads: usize,
    time: usize,
    dim: usize,
    scaling: f32,
    causal: bool,
    out: &mut [f32],
) {
    let qlen = batch * heads * time * dim;
    assert_eq!(query.len(), qlen);
    assert_eq!(key.len(), qlen);
    assert_eq!(value.len(), qlen);
    assert_eq!(out.len(), qlen);

    let mut scores = vec![0.0f32; batch * heads * time * time];
    for b in 0..batch {
        for h in 0..heads {
            for qi in 0..time {
                for kj in 0..time {
                    let mut s = 0.0f32;
                    let qb = ((b * heads + h) * time + qi) * dim;
                    let kb = ((b * heads + h) * time + kj) * dim;
                    for d in 0..dim {
                        s += query[qb + d] * key[kb + d];
                    }
                    s *= scaling;
                    if causal && kj > qi {
                        s = -1.0e9;
                    }
                    scores[(((b * heads + h) * time) + qi) * time + kj] = s;
                }
            }
        }
    }

    let rows = batch * heads * time;
    let mut probs = vec![0.0f32; scores.len()];
    stable_softmax_rows(&scores, rows, time, &mut probs);

    // out_bhqd then transpose to bqhd
    let mut tmp = vec![0.0f32; qlen];
    for b in 0..batch {
        for h in 0..heads {
            for qi in 0..time {
                for d in 0..dim {
                    let mut acc = 0.0f32;
                    for kj in 0..time {
                        let p = probs[(((b * heads + h) * time) + qi) * time + kj];
                        let vb = ((b * heads + h) * time + kj) * dim;
                        acc += p * value[vb + d];
                    }
                    tmp[((b * heads + h) * time + qi) * dim + d] = acc;
                }
            }
        }
    }
    // transpose (B,H,T,D) -> (B,T,H,D)
    for b in 0..batch {
        for t in 0..time {
            for h in 0..heads {
                for d in 0..dim {
                    out[((b * time + t) * heads + h) * dim + d] =
                        tmp[((b * heads + h) * time + t) * dim + d];
                }
            }
        }
    }
}

pub fn gated_mla_output(attn: &[f32], gate_logits: &[f32], out: &mut [f32]) {
    assert_eq!(attn.len(), gate_logits.len());
    assert_eq!(out.len(), attn.len());
    for i in 0..attn.len() {
        let g = 1.0 / (1.0 + (-gate_logits[i]).exp());
        out[i] = attn[i] * g;
    }
}
