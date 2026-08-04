/// L2 normalize over last dim.
pub fn l2_normalize(x: &[f32], rows: usize, dim: usize, eps: f32, out: &mut [f32]) {
    assert_eq!(x.len(), rows * dim);
    assert_eq!(out.len(), x.len());
    for r in 0..rows {
        let mut s = 0.0f32;
        for j in 0..dim {
            let v = x[r * dim + j];
            s += v * v;
        }
        let inv = (s + eps).sqrt().recip();
        for j in 0..dim {
            out[r * dim + j] = x[r * dim + j] * inv;
        }
    }
}

/// Naive recurrent KDA (FLA `naive_recurrent_kda`) with HV == H.
pub fn kda_recurrent(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    batch: usize,
    time: usize,
    heads: usize,
    dim: usize,
    scale: Option<f32>,
    l2norm_qk: bool,
    out: &mut [f32],
    final_state: &mut [f32],
) {
    let scale = scale.unwrap_or((dim as f32).sqrt().recip());
    let nqk = batch * time * heads * dim;
    assert_eq!(q.len(), nqk);
    assert_eq!(k.len(), nqk);
    assert_eq!(v.len(), nqk);
    assert_eq!(g.len(), nqk);
    assert_eq!(beta.len(), batch * time * heads);
    assert_eq!(out.len(), nqk);
    assert_eq!(final_state.len(), batch * heads * dim * dim);

    let mut qq = q.to_vec();
    let mut kk = k.to_vec();
    if l2norm_qk {
        let rows = batch * time * heads;
        let mut tq = vec![0.0f32; qq.len()];
        let mut tk = vec![0.0f32; kk.len()];
        l2_normalize(&qq, rows, dim, 1e-6, &mut tq);
        l2_normalize(&kk, rows, dim, 1e-6, &mut tk);
        qq = tq;
        kk = tk;
    }
    for x in qq.iter_mut() {
        *x *= scale;
    }

    // S: (B, H, K, V) with V == dim
    let mut s = vec![0.0f32; batch * heads * dim * dim];

    for t in 0..time {
        for b in 0..batch {
            for h in 0..heads {
                let q_base = ((b * time + t) * heads + h) * dim;
                let s_base = ((b * heads + h) * dim) * dim; // (K, V) row-major: s[k*dim + v]
                let bval = beta[(b * time + t) * heads + h];

                // S *= exp(g)
                for kd in 0..dim {
                    let eg = g[q_base + kd].exp();
                    for vd in 0..dim {
                        s[s_base + kd * dim + vd] *= eg;
                    }
                }

                // k·S -> (V,)
                let mut ks = vec![0.0f32; dim];
                for vd in 0..dim {
                    let mut acc = 0.0f32;
                    for kd in 0..dim {
                        acc += kk[q_base + kd] * s[s_base + kd * dim + vd];
                    }
                    ks[vd] = acc;
                }

                // delta = v - ks
                let mut delta = vec![0.0f32; dim];
                for vd in 0..dim {
                    delta[vd] = v[q_base + vd] - ks[vd];
                }

                // S += (beta*k) ⊗ delta
                for kd in 0..dim {
                    let bk = bval * kk[q_base + kd];
                    for vd in 0..dim {
                        s[s_base + kd * dim + vd] += bk * delta[vd];
                    }
                }

                // o = q · S
                for vd in 0..dim {
                    let mut acc = 0.0f32;
                    for kd in 0..dim {
                        acc += qq[q_base + kd] * s[s_base + kd * dim + vd];
                    }
                    out[q_base + vd] = acc;
                }
            }
        }
    }
    final_state.copy_from_slice(&s);
}
