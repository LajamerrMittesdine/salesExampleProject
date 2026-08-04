#!/usr/bin/env python3
"""Generate shared golden fixtures for cross-language parity tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "python"))

from kimi_k3_ref import (  # noqa: E402
    apply_attn_res,
    decoder_block_forward,
    expert_ffn,
    gated_mla_output,
    kda_lowerbound_gate,
    kda_recurrent,
    latent_moe_forward,
    l2_normalize,
    mla_eager_attention,
    moe_gate,
    rmsnorm,
    short_conv1d_silu,
    situ_and_mul,
)


def arr(a: np.ndarray) -> list:
    return np.asarray(a, dtype=np.float32).reshape(-1).tolist()


def iarr(a: np.ndarray) -> list:
    return np.asarray(a, dtype=np.int64).reshape(-1).tolist()


def main() -> None:
    rng = np.random.default_rng(20260728)
    fixtures: dict = {"seed": 20260728, "dtype": "float32", "cases": {}}

    # --- rmsnorm ---
    x = rng.standard_normal((2, 3, 8), dtype=np.float32)
    w = rng.standard_normal((8,), dtype=np.float32)
    y = rmsnorm(x, w, eps=1e-5)
    fixtures["cases"]["rmsnorm"] = {
        "shape": list(x.shape),
        "eps": 1e-5,
        "x": arr(x),
        "weight": arr(w),
        "y": arr(y),
    }

    # --- situ ---
    sx = rng.standard_normal((2, 4, 16), dtype=np.float32)
    sy = situ_and_mul(sx, beta=4.0, linear_beta=25.0)
    fixtures["cases"]["situ"] = {
        "shape": list(sx.shape),
        "beta": 4.0,
        "linear_beta": 25.0,
        "x": arr(sx),
        "y": arr(sy),
    }

    # --- moe gate ---
    h = rng.standard_normal((2, 3, 16), dtype=np.float32)
    rw = rng.standard_normal((8, 16), dtype=np.float32) * 0.5
    rb = rng.standard_normal((8,), dtype=np.float32) * 0.1
    idx, wt = moe_gate(h, rw, rb, top_k=2, routed_scaling_factor=1.0, renormalize=True)
    fixtures["cases"]["moe_gate"] = {
        "hidden_shape": list(h.shape),
        "num_experts": 8,
        "top_k": 2,
        "hidden": arr(h),
        "weight": arr(rw),
        "bias": arr(rb),
        "topk_idx": iarr(idx),
        "topk_weight": arr(wt),
    }

    # --- expert ffn ---
    ex = rng.standard_normal((4, 16), dtype=np.float32)
    w1 = rng.standard_normal((8, 16), dtype=np.float32) * 0.2
    w3 = rng.standard_normal((8, 16), dtype=np.float32) * 0.2
    w2 = rng.standard_normal((16, 8), dtype=np.float32) * 0.2
    ey = expert_ffn(ex, w1, w2, w3, beta=4.0, linear_beta=25.0)
    fixtures["cases"]["expert_ffn"] = {
        "x": arr(ex),
        "w1": arr(w1),
        "w2": arr(w2),
        "w3": arr(w3),
        "w1_shape": list(w1.shape),
        "w2_shape": list(w2.shape),
        "w3_shape": list(w3.shape),
        "x_shape": list(ex.shape),
        "y": arr(ey),
        "beta": 4.0,
        "linear_beta": 25.0,
    }

    # --- attn res ---
    n, hdim, nblocks = 3, 8, 2
    prefix = rng.standard_normal((n, hdim), dtype=np.float32)
    bres = rng.standard_normal((n, nblocks, hdim), dtype=np.float32)
    proj = rng.standard_normal((1, hdim), dtype=np.float32)
    nw = np.abs(rng.standard_normal((hdim,), dtype=np.float32)) + 0.5
    ay = apply_attn_res(prefix, bres, proj, nw, eps=1e-5)
    fixtures["cases"]["attn_res"] = {
        "n": n,
        "hidden": hdim,
        "num_blocks": nblocks,
        "prefix_sum": arr(prefix),
        "block_residual": arr(bres),
        "proj_weight": arr(proj),
        "norm_weight": arr(nw),
        "eps": 1e-5,
        "y": arr(ay),
    }

    # --- short conv ---
    cx = rng.standard_normal((1, 8, 4), dtype=np.float32)
    cw = rng.standard_normal((4, 4), dtype=np.float32) * 0.3
    cy = short_conv1d_silu(cx, cw, bias=None)
    fixtures["cases"]["short_conv"] = {
        "x_shape": list(cx.shape),
        "weight_shape": list(cw.shape),
        "x": arr(cx),
        "weight": arr(cw),
        "y": arr(cy),
    }

    # --- kda gate ---
    gg = rng.standard_normal((1, 4, 2, 4), dtype=np.float32)  # B,T,H,K
    alog = rng.uniform(0.0, 2.0, size=(2,)).astype(np.float32)
    dt = rng.standard_normal((2, 4), dtype=np.float32) * 0.1
    gy = kda_lowerbound_gate(gg, alog, dt, lower_bound=-5.0)
    fixtures["cases"]["kda_gate"] = {
        "g_shape": list(gg.shape),
        "g": arr(gg),
        "a_log": arr(alog),
        "dt_bias": arr(dt),
        "lower_bound": -5.0,
        "y": arr(gy),
    }

    # --- kda recurrent ---
    b, tt, hh, kk = 1, 6, 2, 4
    q = rng.standard_normal((b, tt, hh, kk), dtype=np.float32)
    k = rng.standard_normal((b, tt, hh, kk), dtype=np.float32)
    v = rng.standard_normal((b, tt, hh, kk), dtype=np.float32)
    g_raw = rng.standard_normal((b, tt, hh, kk), dtype=np.float32)
    a_log = rng.uniform(0.5, 2.0, size=(hh,)).astype(np.float32)
    dtb = rng.standard_normal((hh, kk), dtype=np.float32) * 0.05
    g = kda_lowerbound_gate(g_raw, a_log, dtb, lower_bound=-5.0)
    beta = 1.0 / (1.0 + np.exp(-rng.standard_normal((b, tt, hh), dtype=np.float32)))
    o, s = kda_recurrent(q, k, v, g, beta, scale=None, l2norm_qk=True)
    fixtures["cases"]["kda_recurrent"] = {
        "q_shape": [b, tt, hh, kk],
        "q": arr(q),
        "k": arr(k),
        "v": arr(v),
        "g_raw": arr(g_raw),
        "a_log": arr(a_log),
        "dt_bias": arr(dtb),
        "beta": arr(beta),
        "lower_bound": -5.0,
        "l2norm_qk": True,
        "o": arr(o),
        "final_state": arr(s),
        "o_shape": list(o.shape),
        "state_shape": list(s.shape),
    }

    # --- mla eager ---
    qb = rng.standard_normal((1, 2, 4, 8), dtype=np.float32)
    kb = rng.standard_normal((1, 2, 4, 8), dtype=np.float32)
    vb = rng.standard_normal((1, 2, 4, 8), dtype=np.float32)
    mo = mla_eager_attention(qb, kb, vb, scaling=8**-0.5, causal=True)
    gate_logits = rng.standard_normal((1, 4, 16), dtype=np.float32)
    flat = mo.reshape(1, 4, 16)
    gated = gated_mla_output(flat, gate_logits)
    fixtures["cases"]["mla_eager"] = {
        "q": arr(qb),
        "k": arr(kb),
        "v": arr(vb),
        "q_shape": list(qb.shape),
        "scaling": float(8**-0.5),
        "attn_out": arr(mo),
        "attn_out_shape": list(mo.shape),
        "gate_logits": arr(gate_logits),
        "gated": arr(gated),
    }

    # --- latent moe (tiny) ---
    hs = rng.standard_normal((1, 2, 16), dtype=np.float32)
    r_w = rng.standard_normal((4, 16), dtype=np.float32) * 0.4
    r_b = rng.standard_normal((4,), dtype=np.float32) * 0.05
    experts = []
    for _ in range(4):
        ew1 = rng.standard_normal((8, 8), dtype=np.float32) * 0.2  # latent 8 -> inter 8
        ew3 = rng.standard_normal((8, 8), dtype=np.float32) * 0.2
        ew2 = rng.standard_normal((8, 8), dtype=np.float32) * 0.2
        experts.append((ew1, ew2, ew3))
    down = rng.standard_normal((8, 16), dtype=np.float32) * 0.2
    up = rng.standard_normal((16, 8), dtype=np.float32) * 0.2
    nrm = np.abs(rng.standard_normal((8,), dtype=np.float32)) + 0.5
    sg = rng.standard_normal((8, 16), dtype=np.float32) * 0.2
    su = rng.standard_normal((8, 16), dtype=np.float32) * 0.2
    sd = rng.standard_normal((16, 8), dtype=np.float32) * 0.2
    moy = latent_moe_forward(
        hs,
        r_w,
        r_b,
        top_k=2,
        expert_weights=experts,
        routed_down=down,
        routed_up=up,
        routed_norm_weight=nrm,
        shared_gate=sg,
        shared_up=su,
        shared_down=sd,
    )
    fixtures["cases"]["latent_moe"] = {
        "hidden_shape": list(hs.shape),
        "hidden": arr(hs),
        "router_weight": arr(r_w),
        "router_bias": arr(r_b),
        "top_k": 2,
        "num_experts": 4,
        "latent": 8,
        "inter": 8,
        "experts_w1": [arr(e[0]) for e in experts],
        "experts_w2": [arr(e[1]) for e in experts],
        "experts_w3": [arr(e[2]) for e in experts],
        "routed_down": arr(down),
        "routed_up": arr(up),
        "routed_norm": arr(nrm),
        "shared_gate": arr(sg),
        "shared_up": arr(su),
        "shared_down": arr(sd),
        "y": arr(moy),
    }

    # --- decoder block residual algebra ---
    dh = rng.standard_normal((1, 2, 8), dtype=np.float32)
    dbr = np.zeros((2, 0, 8), dtype=np.float32)  # N= B*T =2, 0 blocks
    in_w = np.ones((8,), dtype=np.float32)
    post_w = np.ones((8,), dtype=np.float32)
    sap = rng.standard_normal((1, 8), dtype=np.float32)
    san = np.ones((8,), dtype=np.float32)
    mrp = rng.standard_normal((1, 8), dtype=np.float32)
    mrn = np.ones((8,), dtype=np.float32)

    def attn_fn(x):
        return x * np.float32(0.5)

    def mlp_fn(x):
        return x * np.float32(0.25)

    ps, br = decoder_block_forward(
        dh,
        dbr,
        attn_fn,
        mlp_fn,
        in_w,
        post_w,
        sap,
        san,
        mrp,
        mrn,
        layer_idx=0,
        attn_res_block_size=12,
    )
    fixtures["cases"]["decoder_block"] = {
        "hidden": arr(dh),
        "hidden_shape": list(dh.shape),
        "block_residual": arr(dbr),
        "block_residual_shape": list(dbr.shape),
        "input_norm": arr(in_w),
        "post_norm": arr(post_w),
        "self_attn_res_proj": arr(sap),
        "self_attn_res_norm": arr(san),
        "mlp_res_proj": arr(mrp),
        "mlp_res_norm": arr(mrn),
        "layer_idx": 0,
        "attn_res_block_size": 12,
        "attn_scale": 0.5,
        "mlp_scale": 0.25,
        "prefix_sum": arr(ps),
        "prefix_sum_shape": list(ps.shape),
        "out_block_residual": arr(br),
        "out_block_residual_shape": list(br.shape),
    }

    # l2 normalize smoke
    lx = rng.standard_normal((2, 3, 4), dtype=np.float32)
    fixtures["cases"]["l2_normalize"] = {
        "x": arr(lx),
        "x_shape": list(lx.shape),
        "y": arr(l2_normalize(lx)),
        "eps": 1e-6,
    }

    out = ROOT / "fixtures" / "goldens.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixtures, indent=2))
    print(f"Wrote {out} ({out.stat().st_size} bytes), cases={list(fixtures['cases'])}")


if __name__ == "__main__":
    main()
