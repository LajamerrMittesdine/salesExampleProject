/// Stable softmax over the last axis of a flat (rows, cols) matrix, in-place on `x` → `out`.
pub fn stable_softmax_rows(x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    assert_eq!(x.len(), rows * cols);
    assert_eq!(out.len(), rows * cols);
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mut m = f32::NEG_INFINITY;
        for &v in row {
            if v > m {
                m = v;
            }
        }
        let mut sum = 0.0f32;
        for c in 0..cols {
            let e = (row[c] - m).exp();
            out[r * cols + c] = e;
            sum += e;
        }
        for c in 0..cols {
            out[r * cols + c] /= sum;
        }
    }
}

/// `_apply_attn_res` educational port.
pub fn apply_attn_res(
    prefix_sum: &[f32], // (n, h)
    n: usize,
    hidden: usize,
    block_residual: &[f32], // (n, num_blocks, h)
    num_blocks: usize,
    proj_weight: &[f32], // (h,)
    norm_weight: &[f32], // (h,)
    eps: f32,
    out: &mut [f32], // (n, h)
) {
    assert_eq!(prefix_sum.len(), n * hidden);
    assert_eq!(block_residual.len(), n * num_blocks * hidden);
    assert_eq!(proj_weight.len(), hidden);
    assert_eq!(norm_weight.len(), hidden);
    assert_eq!(out.len(), n * hidden);

    let blocks = num_blocks + 1;
    let mut scores = vec![0.0f32; n * blocks];
    let mut probs = vec![0.0f32; n * blocks];
    // We need v for weighted sum — store v as we go or recompute. Store.
    let mut v = vec![0.0f32; n * blocks * hidden];

    for i in 0..n {
        for b in 0..num_blocks {
            for j in 0..hidden {
                v[(i * blocks + b) * hidden + j] =
                    block_residual[(i * num_blocks + b) * hidden + j];
            }
        }
        for j in 0..hidden {
            v[(i * blocks + num_blocks) * hidden + j] = prefix_sum[i * hidden + j];
        }
    }

    for i in 0..n {
        for b in 0..blocks {
            let base = (i * blocks + b) * hidden;
            let mut var = 0.0f32;
            for j in 0..hidden {
                let t = v[base + j];
                var += t * t;
            }
            var /= hidden as f32;
            let inv = (var + eps).sqrt().recip();
            let mut score = 0.0f32;
            for j in 0..hidden {
                let k = v[base + j] * inv;
                score += k * norm_weight[j] * proj_weight[j];
            }
            scores[i * blocks + b] = score;
        }
    }

    stable_softmax_rows(&scores, n, blocks, &mut probs);

    for i in 0..n {
        for j in 0..hidden {
            let mut acc = 0.0f32;
            for b in 0..blocks {
                acc += probs[i * blocks + b] * v[(i * blocks + b) * hidden + j];
            }
            out[i * hidden + j] = acc;
        }
    }
}
