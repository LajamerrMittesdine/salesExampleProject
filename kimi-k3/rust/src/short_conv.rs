#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Causal depthwise short conv + SiLU. `x`: (b,t,c), `weight`: (c,k).
pub fn short_conv1d_silu(
    x: &[f32],
    batch: usize,
    time: usize,
    channels: usize,
    weight: &[f32],
    kernel: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), batch * time * channels);
    assert_eq!(weight.len(), channels * kernel);
    assert_eq!(out.len(), x.len());
    out.fill(0.0);
    for b in 0..batch {
        for t in 0..time {
            for c in 0..channels {
                let mut acc = 0.0f32;
                for i in 0..kernel {
                    let src_t = t as isize - (kernel as isize - 1 - i as isize);
                    let xv = if src_t >= 0 {
                        x[((b * time + src_t as usize) * channels) + c]
                    } else {
                        0.0
                    };
                    // weight[..., 0] multiplies oldest padded sample
                    // For padded index p = t + i in pad scheme (pad left k-1):
                    // y += x_pad[:, i:i+t] * weight[:,:,i]
                    // src at time t uses pad position t+(k-1) for i=k-1 (current)
                    // Our loop: i=0 -> oldest = t-(k-1)
                    acc += xv * weight[c * kernel + i];
                }
                out[(b * time + t) * channels + c] = silu(acc);
            }
        }
    }
}
