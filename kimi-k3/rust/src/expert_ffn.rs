use crate::moe_gate::moe_gate;
use crate::rmsnorm::rmsnorm;
use crate::situ::situ_and_mul;

/// `y = x @ W.T` with W shaped (out, in).
pub fn linear(x: &[f32], rows: usize, in_dim: usize, weight: &[f32], out_dim: usize, out: &mut [f32]) {
    assert_eq!(x.len(), rows * in_dim);
    assert_eq!(weight.len(), out_dim * in_dim);
    assert_eq!(out.len(), rows * out_dim);
    for r in 0..rows {
        for o in 0..out_dim {
            let mut s = 0.0f32;
            let w = &weight[o * in_dim..(o + 1) * in_dim];
            let xr = &x[r * in_dim..(r + 1) * in_dim];
            for j in 0..in_dim {
                s += xr[j] * w[j];
            }
            out[r * out_dim + o] = s;
        }
    }
}

pub fn expert_ffn(
    x: &[f32],
    rows: usize,
    hidden: usize,
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    inter: usize,
    beta: f32,
    linear_beta: Option<f32>,
    out: &mut [f32],
) {
    let mut gate = vec![0.0f32; rows * inter];
    let mut up = vec![0.0f32; rows * inter];
    linear(x, rows, hidden, w1, inter, &mut gate);
    linear(x, rows, hidden, w3, inter, &mut up);
    let mut gate_up = vec![0.0f32; rows * (2 * inter)];
    for r in 0..rows {
        for i in 0..inter {
            gate_up[r * 2 * inter + i] = gate[r * inter + i];
            gate_up[r * 2 * inter + inter + i] = up[r * inter + i];
        }
    }
    let mut act = vec![0.0f32; rows * inter];
    situ_and_mul(&gate_up, 2 * inter, beta, linear_beta, &mut act);
    linear(&act, rows, inter, w2, hidden, out);
}

pub fn dense_mlp(
    x: &[f32],
    rows: usize,
    hidden: usize,
    gate_proj: &[f32],
    up_proj: &[f32],
    down_proj: &[f32],
    inter: usize,
    beta: f32,
    linear_beta: Option<f32>,
    out: &mut [f32],
) {
    expert_ffn(
        x, rows, hidden, gate_proj, down_proj, up_proj, inter, beta, linear_beta, out,
    );
}

pub fn latent_moe_forward(
    hidden_states: &[f32],
    batch: usize,
    seq: usize,
    hidden: usize,
    router_weight: &[f32],
    router_bias: &[f32],
    num_experts: usize,
    top_k: usize,
    experts_w1: &[Vec<f32>],
    experts_w2: &[Vec<f32>],
    experts_w3: &[Vec<f32>],
    latent: usize,
    inter: usize,
    routed_down: &[f32],
    routed_up: &[f32],
    routed_norm: Option<&[f32]>,
    shared_gate: &[f32],
    shared_up: &[f32],
    shared_down: &[f32],
    rms_eps: f32,
    beta: f32,
    linear_beta: Option<f32>,
    out: &mut [f32],
) {
    let n = batch * seq;
    assert_eq!(hidden_states.len(), n * hidden);
    assert_eq!(out.len(), n * hidden);

    let mut topk_idx = vec![0i64; n * top_k];
    let mut topk_weight = vec![0.0f32; n * top_k];
    moe_gate(
        hidden_states,
        n,
        hidden,
        router_weight,
        num_experts,
        router_bias,
        top_k,
        1.0,
        true,
        &mut topk_idx,
        &mut topk_weight,
    );

    let mut latent_x = vec![0.0f32; n * latent];
    linear(hidden_states, n, hidden, routed_down, latent, &mut latent_x);

    let mut y = vec![0.0f32; n * latent];
    for t in 0..n {
        let xt = &latent_x[t * latent..(t + 1) * latent];
        let mut acc = vec![0.0f32; latent];
        for k in 0..top_k {
            let e = topk_idx[t * top_k + k] as usize;
            let mut tmp = vec![0.0f32; latent];
            expert_ffn(
                xt,
                1,
                latent,
                &experts_w1[e],
                &experts_w2[e],
                &experts_w3[e],
                inter,
                beta,
                linear_beta,
                &mut tmp,
            );
            let w = topk_weight[t * top_k + k];
            for j in 0..latent {
                acc[j] += w * tmp[j];
            }
        }
        y[t * latent..(t + 1) * latent].copy_from_slice(&acc);
    }

    if let Some(nw) = routed_norm {
        let mut yn = vec![0.0f32; n * latent];
        rmsnorm(&y, nw, rms_eps, &mut yn);
        y = yn;
    }

    let mut routed = vec![0.0f32; n * hidden];
    linear(&y, n, latent, routed_up, hidden, &mut routed);

    let mut shared = vec![0.0f32; n * hidden];
    dense_mlp(
        hidden_states,
        n,
        hidden,
        shared_gate,
        shared_up,
        shared_down,
        inter,
        beta,
        linear_beta,
        &mut shared,
    );

    for i in 0..n * hidden {
        out[i] = routed[i] + shared[i];
    }
}
