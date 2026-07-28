use kimi_k3_ref::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct GoldenRoot {
    cases: HashMap<String, serde_json::Value>,
}

fn fixtures_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/goldens.json")
}

fn load() -> GoldenRoot {
    let data = fs::read_to_string(fixtures_path()).expect("goldens.json");
    serde_json::from_str(&data).expect("parse goldens")
}

fn f32s(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn i64s(v: &serde_json::Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

fn assert_close(a: &[f32], b: &[f32], atol: f32, rtol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let tol = atol + rtol * y.abs();
        assert!(
            (x - y).abs() <= tol,
            "mismatch at {i}: got {x} expected {y} tol {tol}"
        );
    }
}

#[test]
fn rmsnorm_case() {
    let g = load();
    let c = &g.cases["rmsnorm"];
    let shape: Vec<usize> = c["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    let x = f32s(&c["x"]);
    let w = f32s(&c["weight"]);
    let mut y = vec![0.0f32; x.len()];
    rmsnorm(&x, &w, c["eps"].as_f64().unwrap() as f32, &mut y);
    assert_close(&y, &f32s(&c["y"]), 1e-5, 1e-5);
    assert_eq!(shape.iter().product::<usize>(), x.len());
}

#[test]
fn situ_case() {
    let g = load();
    let c = &g.cases["situ"];
    let shape: Vec<usize> = c["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    let last = *shape.last().unwrap();
    let x = f32s(&c["x"]);
    let mut y = vec![0.0f32; x.len() / 2];
    situ_and_mul(&x, last, 4.0, Some(25.0), &mut y);
    assert_close(&y, &f32s(&c["y"]), 1e-5, 1e-5);
}

#[test]
fn moe_gate_case() {
    let g = load();
    let c = &g.cases["moe_gate"];
    let hs: Vec<usize> = c["hidden_shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    let n = hs[0] * hs[1];
    let h = hs[2];
    let e = c["num_experts"].as_u64().unwrap() as usize;
    let top_k = c["top_k"].as_u64().unwrap() as usize;
    let mut idx = vec![0i64; n * top_k];
    let mut wt = vec![0.0f32; n * top_k];
    moe_gate(
        &f32s(&c["hidden"]),
        n,
        h,
        &f32s(&c["weight"]),
        e,
        &f32s(&c["bias"]),
        top_k,
        1.0,
        true,
        &mut idx,
        &mut wt,
    );
    assert_eq!(idx, i64s(&c["topk_idx"]));
    assert_close(&wt, &f32s(&c["topk_weight"]), 1e-5, 1e-5);
}

#[test]
fn expert_ffn_case() {
    let g = load();
    let c = &g.cases["expert_ffn"];
    let xs: Vec<usize> = c["x_shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    let w1s: Vec<usize> = c["w1_shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    let mut y = vec![0.0f32; xs[0] * xs[1]];
    expert_ffn(
        &f32s(&c["x"]),
        xs[0],
        xs[1],
        &f32s(&c["w1"]),
        &f32s(&c["w2"]),
        &f32s(&c["w3"]),
        w1s[0],
        4.0,
        Some(25.0),
        &mut y,
    );
    assert_close(&y, &f32s(&c["y"]), 1e-4, 1e-4);
}

#[test]
fn attn_res_case() {
    let g = load();
    let c = &g.cases["attn_res"];
    let n = c["n"].as_u64().unwrap() as usize;
    let h = c["hidden"].as_u64().unwrap() as usize;
    let nb = c["num_blocks"].as_u64().unwrap() as usize;
    let mut y = vec![0.0f32; n * h];
    apply_attn_res(
        &f32s(&c["prefix_sum"]),
        n,
        h,
        &f32s(&c["block_residual"]),
        nb,
        &f32s(&c["proj_weight"]),
        &f32s(&c["norm_weight"]),
        1e-5,
        &mut y,
    );
    assert_close(&y, &f32s(&c["y"]), 1e-5, 1e-5);
}

#[test]
fn short_conv_case() {
    let g = load();
    let c = &g.cases["short_conv"];
    let xs: Vec<usize> = c["x_shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    let ws: Vec<usize> = c["weight_shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    let mut y = vec![0.0f32; xs[0] * xs[1] * xs[2]];
    short_conv1d_silu(&f32s(&c["x"]), xs[0], xs[1], xs[2], &f32s(&c["weight"]), ws[1], &mut y);
    assert_close(&y, &f32s(&c["y"]), 1e-5, 1e-5);
}

#[test]
fn kda_gate_case() {
    let g = load();
    let c = &g.cases["kda_gate"];
    let gs: Vec<usize> = c["g_shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    let outer = gs[0] * gs[1];
    let mut y = vec![0.0f32; f32s(&c["g"]).len()];
    let dt = f32s(&c["dt_bias"]);
    kda_lowerbound_gate(
        &f32s(&c["g"]),
        outer,
        gs[2],
        gs[3],
        &f32s(&c["a_log"]),
        Some(&dt),
        -5.0,
        &mut y,
    );
    assert_close(&y, &f32s(&c["y"]), 1e-5, 1e-5);
}

#[test]
fn kda_recurrent_case() {
    let g = load();
    let c = &g.cases["kda_recurrent"];
    let qs: Vec<usize> = c["q_shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    let (b, t, h, d) = (qs[0], qs[1], qs[2], qs[3]);
    let mut g_out = vec![0.0f32; b * t * h * d];
    let dt = f32s(&c["dt_bias"]);
    kda_lowerbound_gate(
        &f32s(&c["g_raw"]),
        b * t,
        h,
        d,
        &f32s(&c["a_log"]),
        Some(&dt),
        -5.0,
        &mut g_out,
    );
    let mut o = vec![0.0f32; b * t * h * d];
    let mut s = vec![0.0f32; b * h * d * d];
    kda_recurrent(
        &f32s(&c["q"]),
        &f32s(&c["k"]),
        &f32s(&c["v"]),
        &g_out,
        &f32s(&c["beta"]),
        b,
        t,
        h,
        d,
        None,
        true,
        &mut o,
        &mut s,
    );
    assert_close(&o, &f32s(&c["o"]), 1e-4, 1e-4);
    assert_close(&s, &f32s(&c["final_state"]), 1e-4, 1e-4);
}

#[test]
fn mla_eager_case() {
    let g = load();
    let c = &g.cases["mla_eager"];
    let qs: Vec<usize> = c["q_shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    let (b, h, t, d) = (qs[0], qs[1], qs[2], qs[3]);
    let mut o = vec![0.0f32; b * t * h * d];
    mla_eager_attention(
        &f32s(&c["q"]),
        &f32s(&c["k"]),
        &f32s(&c["v"]),
        b,
        h,
        t,
        d,
        c["scaling"].as_f64().unwrap() as f32,
        true,
        &mut o,
    );
    assert_close(&o, &f32s(&c["attn_out"]), 1e-4, 1e-4);
    let mut gated = vec![0.0f32; o.len()];
    gated_mla_output(&o, &f32s(&c["gate_logits"]), &mut gated);
    assert_close(&gated, &f32s(&c["gated"]), 1e-4, 1e-4);
}

#[test]
fn latent_moe_case() {
    let g = load();
    let c = &g.cases["latent_moe"];
    let hs: Vec<usize> = c["hidden_shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    let ne = c["num_experts"].as_u64().unwrap() as usize;
    let latent = c["latent"].as_u64().unwrap() as usize;
    let inter = c["inter"].as_u64().unwrap() as usize;
    let mut ew1 = Vec::new();
    let mut ew2 = Vec::new();
    let mut ew3 = Vec::new();
    for i in 0..ne {
        ew1.push(f32s(&c["experts_w1"][i]));
        ew2.push(f32s(&c["experts_w2"][i]));
        ew3.push(f32s(&c["experts_w3"][i]));
    }
    let mut y = vec![0.0f32; hs[0] * hs[1] * hs[2]];
    let nrm = f32s(&c["routed_norm"]);
    latent_moe_forward(
        &f32s(&c["hidden"]),
        hs[0],
        hs[1],
        hs[2],
        &f32s(&c["router_weight"]),
        &f32s(&c["router_bias"]),
        ne,
        c["top_k"].as_u64().unwrap() as usize,
        &ew1,
        &ew2,
        &ew3,
        latent,
        inter,
        &f32s(&c["routed_down"]),
        &f32s(&c["routed_up"]),
        Some(&nrm),
        &f32s(&c["shared_gate"]),
        &f32s(&c["shared_up"]),
        &f32s(&c["shared_down"]),
        1e-5,
        4.0,
        Some(25.0),
        &mut y,
    );
    assert_close(&y, &f32s(&c["y"]), 1e-4, 1e-4);
}

#[test]
fn decoder_block_case() {
    let g = load();
    let c = &g.cases["decoder_block"];
    let hs: Vec<usize> = c["hidden_shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    let brs: Vec<usize> = c["block_residual_shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    let mut prefix = vec![0.0f32; hs[0] * hs[1] * hs[2]];
    let mut br_out = Vec::new();
    let mut nb_out = 0usize;
    decoder_block_forward(
        &f32s(&c["hidden"]),
        hs[0],
        hs[1],
        hs[2],
        &f32s(&c["block_residual"]),
        brs[1],
        &f32s(&c["input_norm"]),
        &f32s(&c["post_norm"]),
        &f32s(&c["self_attn_res_proj"]),
        &f32s(&c["self_attn_res_norm"]),
        &f32s(&c["mlp_res_proj"]),
        &f32s(&c["mlp_res_norm"]),
        c["layer_idx"].as_u64().unwrap() as usize,
        c["attn_res_block_size"].as_u64().unwrap() as usize,
        c["attn_scale"].as_f64().unwrap() as f32,
        c["mlp_scale"].as_f64().unwrap() as f32,
        1e-5,
        &mut prefix,
        &mut br_out,
        &mut nb_out,
    );
    assert_close(&prefix, &f32s(&c["prefix_sum"]), 1e-4, 1e-4);
    assert_close(&br_out, &f32s(&c["out_block_residual"]), 1e-4, 1e-4);
}

#[test]
fn l2_normalize_case() {
    let g = load();
    let c = &g.cases["l2_normalize"];
    let xs: Vec<usize> = c["x_shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    let dim = *xs.last().unwrap();
    let rows = xs.iter().product::<usize>() / dim;
    let mut y = vec![0.0f32; rows * dim];
    l2_normalize(&f32s(&c["x"]), rows, dim, 1e-6, &mut y);
    assert_close(&y, &f32s(&c["y"]), 1e-5, 1e-5);
}
