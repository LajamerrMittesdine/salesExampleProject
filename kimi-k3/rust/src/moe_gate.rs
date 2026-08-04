/// MoE gate with deterministic top-k (educational).

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `hidden`: (n, h), `weight`: (e, h), `bias`: (e,)
/// Writes flat `topk_idx` / `topk_weight` of length `n * top_k`.
pub fn moe_gate(
    hidden: &[f32],
    n: usize,
    h: usize,
    weight: &[f32],
    num_experts: usize,
    bias: &[f32],
    top_k: usize,
    routed_scaling_factor: f32,
    renormalize: bool,
    topk_idx: &mut [i64],
    topk_weight: &mut [f32],
) {
    assert_eq!(hidden.len(), n * h);
    assert_eq!(weight.len(), num_experts * h);
    assert_eq!(bias.len(), num_experts);
    assert_eq!(topk_idx.len(), n * top_k);
    assert_eq!(topk_weight.len(), n * top_k);

    let mut scores = vec![0.0f32; num_experts];
    let mut choice = vec![0.0f32; num_experts];

    for row in 0..n {
        let x = &hidden[row * h..(row + 1) * h];
        for e in 0..num_experts {
            let mut logit = 0.0f32;
            let wrow = &weight[e * h..(e + 1) * h];
            for j in 0..h {
                logit += x[j] * wrow[j];
            }
            scores[e] = sigmoid(logit);
            choice[e] = scores[e] + bias[e];
        }
        let mut order: Vec<usize> = (0..num_experts).collect();
        order.sort_by(|&a, &b| {
            choice[b]
                .partial_cmp(&choice[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let mut chosen: Vec<usize> = order.into_iter().take(top_k).collect();
        chosen.sort_unstable();
        for (k, &ei) in chosen.iter().enumerate() {
            topk_idx[row * top_k + k] = ei as i64;
            topk_weight[row * top_k + k] = scores[ei];
        }
        if top_k > 1 && renormalize {
            let mut denom = 1e-20f32;
            for k in 0..top_k {
                denom += topk_weight[row * top_k + k];
            }
            for k in 0..top_k {
                topk_weight[row * top_k + k] /= denom;
            }
        }
        for k in 0..top_k {
            topk_weight[row * top_k + k] *= routed_scaling_factor;
        }
    }
}
