#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// K3 lower-bound gate: `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`.
/// `g`: flat (..., h, k) with leading size `outer`, `a_log`: (h,), `dt_bias`: (h,k)
pub fn kda_lowerbound_gate(
    g: &[f32],
    outer: usize,
    heads: usize,
    dim: usize,
    a_log: &[f32],
    dt_bias: Option<&[f32]>,
    lower_bound: f32,
    out: &mut [f32],
) {
    assert_eq!(g.len(), outer * heads * dim);
    assert_eq!(out.len(), g.len());
    assert_eq!(a_log.len(), heads);
    let mut a_exp = vec![0.0f32; heads];
    for h in 0..heads {
        a_exp[h] = a_log[h].exp();
    }
    for o in 0..outer {
        for h in 0..heads {
            for d in 0..dim {
                let idx = (o * heads + h) * dim + d;
                let mut v = g[idx];
                if let Some(dt) = dt_bias {
                    v += dt[h * dim + d];
                }
                out[idx] = lower_bound * sigmoid(a_exp[h] * v);
            }
        }
    }
}
