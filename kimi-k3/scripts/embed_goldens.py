#!/usr/bin/env python3
"""Regenerate fixtures/goldens_embedded.h from fixtures/goldens.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fmt(v: float) -> str:
    return f"{float(v):.9e}f"


def emit_f32(out: list[str], name: str, arr: list) -> None:
    out.append(f"static const float {name}[] = {{")
    line: list[str] = []
    for v in arr:
        line.append(fmt(v))
        if len(line) == 8:
            out.append("  " + ", ".join(line) + ",")
            line = []
    if line:
        out.append("  " + ", ".join(line) + ",")
    if not arr:
        out.append("  0.0e+00f, /* empty placeholder */")
    out.append("};")
    out.append(f"static const size_t {name}_len = {len(arr)};")


def emit_i64(out: list[str], name: str, arr: list) -> None:
    out.append(f"static const long long {name}[] = {{")
    line: list[str] = []
    for v in arr:
        line.append(str(int(v)))
        if len(line) == 16:
            out.append("  " + ", ".join(line) + ",")
            line = []
    if line:
        out.append("  " + ", ".join(line) + ",")
    if not arr:
        out.append("  0,")
    out.append("};")
    out.append(f"static const size_t {name}_len = {len(arr)};")


def main() -> None:
    g = json.loads((ROOT / "fixtures" / "goldens.json").read_text())
    out: list[str] = [
        "/* Auto-generated from fixtures/goldens.json — do not edit */",
        "#pragma once",
        "#include <stddef.h>",
    ]
    cases = g["cases"]

    c = cases["rmsnorm"]
    emit_f32(out, "g_rmsnorm_x", c["x"])
    emit_f32(out, "g_rmsnorm_w", c["weight"])
    emit_f32(out, "g_rmsnorm_y", c["y"])
    out.append(f"static const float g_rmsnorm_eps = {fmt(c['eps'])};")
    out.append(
        f"static const size_t g_rmsnorm_shape[3] = {{{c['shape'][0]}, {c['shape'][1]}, {c['shape'][2]}}};"
    )

    c = cases["situ"]
    emit_f32(out, "g_situ_x", c["x"])
    emit_f32(out, "g_situ_y", c["y"])
    out.append(
        f"static const size_t g_situ_shape[3] = {{{c['shape'][0]}, {c['shape'][1]}, {c['shape'][2]}}};"
    )

    c = cases["moe_gate"]
    emit_f32(out, "g_moe_hidden", c["hidden"])
    emit_f32(out, "g_moe_weight", c["weight"])
    emit_f32(out, "g_moe_bias", c["bias"])
    emit_i64(out, "g_moe_idx", c["topk_idx"])
    emit_f32(out, "g_moe_wt", c["topk_weight"])
    hs = c["hidden_shape"]
    out.append(
        f"static const size_t g_moe_B={hs[0]}, g_moe_S={hs[1]}, g_moe_H={hs[2]}, g_moe_E={c['num_experts']}, g_moe_K={c['top_k']};"
    )

    c = cases["expert_ffn"]
    emit_f32(out, "g_effn_x", c["x"])
    emit_f32(out, "g_effn_w1", c["w1"])
    emit_f32(out, "g_effn_w2", c["w2"])
    emit_f32(out, "g_effn_w3", c["w3"])
    emit_f32(out, "g_effn_y", c["y"])
    out.append(
        f"static const size_t g_effn_rows={c['x_shape'][0]}, g_effn_h={c['x_shape'][1]}, g_effn_inter={c['w1_shape'][0]};"
    )

    c = cases["attn_res"]
    emit_f32(out, "g_ar_prefix", c["prefix_sum"])
    emit_f32(out, "g_ar_block", c["block_residual"])
    emit_f32(out, "g_ar_proj", c["proj_weight"])
    emit_f32(out, "g_ar_norm", c["norm_weight"])
    emit_f32(out, "g_ar_y", c["y"])
    out.append(
        f"static const size_t g_ar_n={c['n']}, g_ar_h={c['hidden']}, g_ar_nb={c['num_blocks']};"
    )

    c = cases["short_conv"]
    emit_f32(out, "g_sc_x", c["x"])
    emit_f32(out, "g_sc_w", c["weight"])
    emit_f32(out, "g_sc_y", c["y"])
    out.append(
        f"static const size_t g_sc_b={c['x_shape'][0]}, g_sc_t={c['x_shape'][1]}, g_sc_c={c['x_shape'][2]}, g_sc_k={c['weight_shape'][1]};"
    )

    c = cases["kda_gate"]
    emit_f32(out, "g_kg_g", c["g"])
    emit_f32(out, "g_kg_alog", c["a_log"])
    emit_f32(out, "g_kg_dt", c["dt_bias"])
    emit_f32(out, "g_kg_y", c["y"])
    gs = c["g_shape"]
    out.append(
        f"static const size_t g_kg_B={gs[0]}, g_kg_T={gs[1]}, g_kg_H={gs[2]}, g_kg_D={gs[3]};"
    )

    c = cases["kda_recurrent"]
    for name in ["q", "k", "v", "g_raw", "a_log", "dt_bias", "beta", "o", "final_state"]:
        emit_f32(out, "g_kda_" + name, c[name])
    qs = c["q_shape"]
    out.append(
        f"static const size_t g_kda_B={qs[0]}, g_kda_T={qs[1]}, g_kda_H={qs[2]}, g_kda_D={qs[3]};"
    )

    c = cases["mla_eager"]
    for name in ["q", "k", "v", "attn_out", "gate_logits", "gated"]:
        emit_f32(out, "g_mla_" + name, c[name])
    qs = c["q_shape"]
    out.append(
        f"static const size_t g_mla_B={qs[0]}, g_mla_H={qs[1]}, g_mla_T={qs[2]}, g_mla_D={qs[3]};"
    )
    out.append(f"static const float g_mla_scale = {fmt(c['scaling'])};")

    c = cases["latent_moe"]
    emit_f32(out, "g_lm_hidden", c["hidden"])
    emit_f32(out, "g_lm_rw", c["router_weight"])
    emit_f32(out, "g_lm_rb", c["router_bias"])
    emit_f32(out, "g_lm_down", c["routed_down"])
    emit_f32(out, "g_lm_up", c["routed_up"])
    emit_f32(out, "g_lm_norm", c["routed_norm"])
    emit_f32(out, "g_lm_sg", c["shared_gate"])
    emit_f32(out, "g_lm_su", c["shared_up"])
    emit_f32(out, "g_lm_sd", c["shared_down"])
    emit_f32(out, "g_lm_y", c["y"])
    hs = c["hidden_shape"]
    out.append(
        f"static const size_t g_lm_B={hs[0]}, g_lm_S={hs[1]}, g_lm_H={hs[2]}, g_lm_E={c['num_experts']}, g_lm_K={c['top_k']}, g_lm_L={c['latent']}, g_lm_I={c['inter']};"
    )
    for i in range(c["num_experts"]):
        emit_f32(out, f"g_lm_e{i}_w1", c["experts_w1"][i])
        emit_f32(out, f"g_lm_e{i}_w2", c["experts_w2"][i])
        emit_f32(out, f"g_lm_e{i}_w3", c["experts_w3"][i])

    c = cases["decoder_block"]
    mapping = [
        ("hidden", "hidden"),
        ("br", "block_residual"),
        ("in", "input_norm"),
        ("post", "post_norm"),
        ("sap", "self_attn_res_proj"),
        ("san", "self_attn_res_norm"),
        ("mrp", "mlp_res_proj"),
        ("mrn", "mlp_res_norm"),
        ("ps", "prefix_sum"),
        ("obr", "out_block_residual"),
    ]
    for name, key in mapping:
        emit_f32(out, "g_db_" + name, c[key])
    hs = c["hidden_shape"]
    brs = c["block_residual_shape"]
    out.append(
        f"static const size_t g_db_B={hs[0]}, g_db_S={hs[1]}, g_db_H={hs[2]}, g_db_NB={brs[1]};"
    )
    out.append(
        f"static const size_t g_db_layer={c['layer_idx']}, g_db_bs={c['attn_res_block_size']};"
    )
    out.append(
        f"static const float g_db_attn={fmt(c['attn_scale'])}, g_db_mlp={fmt(c['mlp_scale'])};"
    )

    c = cases["l2_normalize"]
    emit_f32(out, "g_l2_x", c["x"])
    emit_f32(out, "g_l2_y", c["y"])
    xs = c["x_shape"]
    out.append(f"static const size_t g_l2_shape[3] = {{{xs[0]}, {xs[1]}, {xs[2]}}};")

    path = ROOT / "fixtures" / "goldens_embedded.h"
    path.write_text("\n".join(out) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
