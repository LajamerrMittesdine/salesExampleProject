/// SiTU-GLU matching `SituAndMul`.
///
/// `x` is flattened `(..., last_dim)` with even `last_dim`; `out` is `(..., last_dim/2)`.
pub fn situ_and_mul(x: &[f32], last_dim: usize, beta: f32, linear_beta: Option<f32>, out: &mut [f32]) {
    assert!(last_dim % 2 == 0, "SiTU expects even last dim");
    let d = last_dim / 2;
    assert_eq!(x.len() % last_dim, 0);
    assert_eq!(out.len(), x.len() / 2);
    let rows = x.len() / last_dim;
    for r in 0..rows {
        let xs = &x[r * last_dim..(r + 1) * last_dim];
        let os = &mut out[r * d..(r + 1) * d];
        for i in 0..d {
            let gate = xs[i];
            let mut up = xs[d + i];
            let situ_a = beta * (gate / beta).tanh() * sigmoid(gate);
            if let Some(lb) = linear_beta {
                up = lb * (up / lb).tanh();
            }
            os[i] = situ_a * up;
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
