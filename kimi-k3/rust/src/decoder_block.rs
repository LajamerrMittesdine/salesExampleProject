use crate::attn_res::apply_attn_res;
use crate::rmsnorm::rmsnorm;

/// Educational AttnRes decoder-block residual algebra.
/// `attn_scale` / `mlp_scale` stand in for full attention/MoE (fixture uses constants).
pub fn decoder_block_forward(
    hidden_states: &[f32],
    batch: usize,
    seq: usize,
    hidden: usize,
    block_residual: &[f32],
    num_blocks: usize,
    input_norm: &[f32],
    post_norm: &[f32],
    self_attn_res_proj: &[f32],
    self_attn_res_norm: &[f32],
    mlp_res_proj: &[f32],
    mlp_res_norm: &[f32],
    layer_idx: usize,
    attn_res_block_size: usize,
    attn_scale: f32,
    mlp_scale: f32,
    rms_eps: f32,
    prefix_out: &mut [f32],
    block_residual_out: &mut Vec<f32>,
    num_blocks_out: &mut usize,
) {
    let n = batch * seq;
    assert_eq!(hidden_states.len(), n * hidden);
    assert_eq!(prefix_out.len(), n * hidden);

    let mut hidden_buf = hidden_states.to_vec();
    let mut prefix: Option<Vec<f32>> = Some(hidden_states.to_vec());
    let mut br = block_residual.to_vec();
    let mut nb = num_blocks;

    if nb > 0 {
        let mut tmp = vec![0.0f32; n * hidden];
        apply_attn_res(
            prefix.as_ref().unwrap(),
            n,
            hidden,
            &br,
            nb,
            self_attn_res_proj,
            self_attn_res_norm,
            rms_eps,
            &mut tmp,
        );
        hidden_buf = tmp;
    }

    if layer_idx % attn_res_block_size == 0 {
        // append prefix as new block
        let p = prefix.as_ref().unwrap();
        br.extend_from_slice(p);
        nb += 1;
        prefix = None;
    }

    let mut normed = vec![0.0f32; n * hidden];
    rmsnorm(&hidden_buf, input_norm, rms_eps, &mut normed);
    for x in normed.iter_mut() {
        *x *= attn_scale;
    }
    let attn_out = normed;

    let mut prefix_sum = if let Some(p) = prefix {
        let mut s = p;
        for i in 0..n * hidden {
            s[i] += attn_out[i];
        }
        s
    } else {
        attn_out.clone()
    };

    let mut hidden2 = vec![0.0f32; n * hidden];
    apply_attn_res(
        &prefix_sum,
        n,
        hidden,
        &br,
        nb,
        mlp_res_proj,
        mlp_res_norm,
        rms_eps,
        &mut hidden2,
    );

    let mut post = vec![0.0f32; n * hidden];
    rmsnorm(&hidden2, post_norm, rms_eps, &mut post);
    for x in post.iter_mut() {
        *x *= mlp_scale;
    }

    for i in 0..n * hidden {
        prefix_sum[i] += post[i];
    }

    prefix_out.copy_from_slice(&prefix_sum);
    *block_residual_out = br;
    *num_blocks_out = nb;
}
