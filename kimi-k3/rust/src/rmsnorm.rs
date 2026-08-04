/// RMSNorm matching `KimiRMSNorm`.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    let hidden = weight.len();
    assert!(hidden > 0 && x.len() % hidden == 0);
    let rows = x.len() / hidden;
    for r in 0..rows {
        let base = r * hidden;
        let mut var = 0.0f32;
        for j in 0..hidden {
            let v = x[base + j];
            var += v * v;
        }
        var /= hidden as f32;
        let inv = (var + eps).sqrt().recip();
        for j in 0..hidden {
            out[base + j] = x[base + j] * inv * weight[j];
        }
    }
}
