"""Verify reference ops match committed golden fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

GOLDEN = json.loads((ROOT / "fixtures" / "goldens.json").read_text())


def _f(case, key, shape=None):
    a = np.asarray(case[key], dtype=np.float32)
    if shape is not None:
        a = a.reshape(shape)
    return a


def _assert_close(a, b, atol=1e-5, rtol=1e-5):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def test_rmsnorm():
    c = GOLDEN["cases"]["rmsnorm"]
    x = _f(c, "x", c["shape"])
    w = _f(c, "weight")
    y = rmsnorm(x, w, eps=c["eps"])
    _assert_close(y.reshape(-1), _f(c, "y"))


def test_situ():
    c = GOLDEN["cases"]["situ"]
    x = _f(c, "x", c["shape"])
    y = situ_and_mul(x, beta=c["beta"], linear_beta=c["linear_beta"])
    _assert_close(y.reshape(-1), _f(c, "y"))


def test_moe_gate():
    c = GOLDEN["cases"]["moe_gate"]
    h = _f(c, "hidden", c["hidden_shape"])
    w = _f(c, "weight", (c["num_experts"], c["hidden_shape"][-1]))
    b = _f(c, "bias")
    idx, wt = moe_gate(h, w, b, top_k=c["top_k"])
    assert np.array_equal(idx.reshape(-1), np.asarray(c["topk_idx"], dtype=np.int64))
    _assert_close(wt.reshape(-1), _f(c, "topk_weight"))


def test_expert_ffn():
    c = GOLDEN["cases"]["expert_ffn"]
    y = expert_ffn(
        _f(c, "x", c["x_shape"]),
        _f(c, "w1", c["w1_shape"]),
        _f(c, "w2", c["w2_shape"]),
        _f(c, "w3", c["w3_shape"]),
        beta=c["beta"],
        linear_beta=c["linear_beta"],
    )
    _assert_close(y.reshape(-1), _f(c, "y"), atol=1e-4, rtol=1e-4)


def test_attn_res():
    c = GOLDEN["cases"]["attn_res"]
    y = apply_attn_res(
        _f(c, "prefix_sum", (c["n"], c["hidden"])),
        _f(c, "block_residual", (c["n"], c["num_blocks"], c["hidden"])),
        _f(c, "proj_weight", (1, c["hidden"])),
        _f(c, "norm_weight"),
        eps=c["eps"],
    )
    _assert_close(y.reshape(-1), _f(c, "y"))


def test_short_conv():
    c = GOLDEN["cases"]["short_conv"]
    y = short_conv1d_silu(_f(c, "x", c["x_shape"]), _f(c, "weight", c["weight_shape"]))
    _assert_close(y.reshape(-1), _f(c, "y"))


def test_kda_gate():
    c = GOLDEN["cases"]["kda_gate"]
    y = kda_lowerbound_gate(
        _f(c, "g", c["g_shape"]),
        _f(c, "a_log"),
        _f(c, "dt_bias", (c["g_shape"][-2], c["g_shape"][-1])),
        lower_bound=c["lower_bound"],
    )
    _assert_close(y.reshape(-1), _f(c, "y"))


def test_kda_recurrent():
    c = GOLDEN["cases"]["kda_recurrent"]
    shp = c["q_shape"]
    g = kda_lowerbound_gate(
        _f(c, "g_raw", shp),
        _f(c, "a_log"),
        _f(c, "dt_bias", (shp[2], shp[3])),
        lower_bound=c["lower_bound"],
    )
    o, s = kda_recurrent(
        _f(c, "q", shp),
        _f(c, "k", shp),
        _f(c, "v", shp),
        g,
        _f(c, "beta", shp[:3]),
        l2norm_qk=c["l2norm_qk"],
    )
    _assert_close(o.reshape(-1), _f(c, "o"), atol=1e-4, rtol=1e-4)
    _assert_close(s.reshape(-1), _f(c, "final_state"), atol=1e-4, rtol=1e-4)


def test_mla_eager():
    c = GOLDEN["cases"]["mla_eager"]
    mo = mla_eager_attention(
        _f(c, "q", c["q_shape"]),
        _f(c, "k", c["q_shape"]),
        _f(c, "v", c["q_shape"]),
        scaling=c["scaling"],
        causal=True,
    )
    _assert_close(mo.reshape(-1), _f(c, "attn_out"), atol=1e-4, rtol=1e-4)
    gated = gated_mla_output(mo.reshape(1, 4, 16), _f(c, "gate_logits", (1, 4, 16)))
    _assert_close(gated.reshape(-1), _f(c, "gated"), atol=1e-4, rtol=1e-4)


def test_latent_moe():
    c = GOLDEN["cases"]["latent_moe"]
    experts = [
        (
            _f(c, "experts_w1")[i].reshape(c["inter"], c["latent"])
            if False
            else np.asarray(c["experts_w1"][i], dtype=np.float32).reshape(c["inter"], c["latent"]),
            np.asarray(c["experts_w2"][i], dtype=np.float32).reshape(c["latent"], c["inter"]),
            np.asarray(c["experts_w3"][i], dtype=np.float32).reshape(c["inter"], c["latent"]),
        )
        for i in range(c["num_experts"])
    ]
    y = latent_moe_forward(
        _f(c, "hidden", c["hidden_shape"]),
        _f(c, "router_weight", (c["num_experts"], c["hidden_shape"][-1])),
        _f(c, "router_bias"),
        top_k=c["top_k"],
        expert_weights=experts,
        routed_down=_f(c, "routed_down", (c["latent"], c["hidden_shape"][-1])),
        routed_up=_f(c, "routed_up", (c["hidden_shape"][-1], c["latent"])),
        routed_norm_weight=_f(c, "routed_norm"),
        shared_gate=_f(c, "shared_gate", (c["inter"], c["hidden_shape"][-1])),
        shared_up=_f(c, "shared_up", (c["inter"], c["hidden_shape"][-1])),
        shared_down=_f(c, "shared_down", (c["hidden_shape"][-1], c["inter"])),
    )
    _assert_close(y.reshape(-1), _f(c, "y"), atol=1e-4, rtol=1e-4)


def test_decoder_block():
    c = GOLDEN["cases"]["decoder_block"]
    hs = _f(c, "hidden", c["hidden_shape"])
    br = _f(c, "block_residual", c["block_residual_shape"])

    def attn_fn(x):
        return x * np.float32(c["attn_scale"])

    def mlp_fn(x):
        return x * np.float32(c["mlp_scale"])

    ps, obr = decoder_block_forward(
        hs,
        br,
        attn_fn,
        mlp_fn,
        _f(c, "input_norm"),
        _f(c, "post_norm"),
        _f(c, "self_attn_res_proj", (1, c["hidden_shape"][-1])),
        _f(c, "self_attn_res_norm"),
        _f(c, "mlp_res_proj", (1, c["hidden_shape"][-1])),
        _f(c, "mlp_res_norm"),
        layer_idx=c["layer_idx"],
        attn_res_block_size=c["attn_res_block_size"],
    )
    _assert_close(ps.reshape(-1), _f(c, "prefix_sum"), atol=1e-4, rtol=1e-4)
    _assert_close(obr.reshape(-1), _f(c, "out_block_residual"), atol=1e-4, rtol=1e-4)


def test_l2_normalize():
    c = GOLDEN["cases"]["l2_normalize"]
    y = l2_normalize(_f(c, "x", c["x_shape"]), eps=c["eps"])
    _assert_close(y.reshape(-1), _f(c, "y"))
