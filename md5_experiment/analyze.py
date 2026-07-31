"""Plotting and scientific summary helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def plot_history(history: list[dict], metrics: list[tuple[str, str]], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    for ax, (key, label) in zip(axes[0], metrics):
        ax.plot(epochs, [h[key] for h in history], marker="o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def scientific_verdict(report: dict[str, Any]) -> str:
    lines = [
        "MD5 Learning Experiment — Scientific Summary",
        "=" * 48,
        "",
        "Hypothesis: a neural net can learn MD5's structure well enough to",
        "generalize to unseen plaintexts and accelerate preimage search.",
        "",
    ]

    if "memorization" in report:
        mem = report["memorization"]
        lines += [
            "Memorization probe (fit tiny train subset, test transfer):",
            f"  subset exact match   : {mem['memorized_subset_exact_match']:.4f}",
            f"  held-out exact match : {mem['held_out_exact_match']:.6f}  "
            f"(chance ≈ {mem['random_exact_match']:.6f})",
            "",
        ]

    if "forward" in report:
        fwd = report["forward"]["test"]
        lines += [
            "Forward model (plain → hash bits) on held-out test:",
            f"  bit accuracy : {fwd['bit_accuracy']:.4f}  (chance ≈ 0.50)",
            f"  exact match  : {fwd['exact_match']:.6f}  (chance ≈ 0)",
        ]
        if "avalanche" in report:
            av = report["avalanche"]
            lines += [
                f"  true avalanche   : {av['true_mean_bit_flip_fraction']:.4f}",
                f"  model avalanche  : {av['pred_mean_bit_flip_fraction']:.4f}",
            ]
        lines.append("")

    if "inverse" in report:
        inv = report["inverse"]
        tr, te = inv["train"], inv["test"]
        base = inv["baselines"]
        lines += [
            "Inverse model (hash → plain) :",
            f"  train exact match : {tr['exact_match']:.4f}",
            f"  test  exact match : {te['exact_match']:.6f}  "
            f"(chance ≈ {base['random_exact_match']:.6f})",
            f"  test  char acc    : {te['char_accuracy']:.4f}  "
            f"(chance ≈ {base['random_char_accuracy']:.4f})",
        ]
        if "search" in report:
            s = report["search"]
            lines += [
                "Search acceleration probe (rank of true plaintext):",
                f"  mean rank        : {s['mean_rank']:.1f}",
                f"  random mean rank : {s['expected_random_mean_rank']:.1f}",
                f"  top-1 hit rate   : {s['top1_hit_rate']:.4f}",
            ]
        lines.append("")

    # Compact verdict
    can_memorize = False
    generalization = False
    if "memorization" in report:
        can_memorize = report["memorization"]["memorized_subset_exact_match"] >= 0.95
    if "inverse" in report:
        te = report["inverse"]["test"]["exact_match"]
        chance = report["inverse"]["baselines"]["random_exact_match"]
        if te > 20 * chance:
            generalization = True
    if "memorization" in report:
        te_m = report["memorization"]["held_out_exact_match"]
        chance_m = report["memorization"]["random_exact_match"]
        if te_m > 20 * chance_m:
            generalization = True

    lines.append("Verdict:")
    if generalization:
        lines.append(
            "  Model shows above-chance held-out recovery — investigate capacity,"
            " leakage, and whether the constrained space is too small."
        )
    elif can_memorize:
        lines.append(
            "  Nets can memorize small MD5 lookup tables, but held-out recovery"
            " stays near chance: no evidence of learning an invertible MD5 structure"
            " or a search-accelerating flaw."
        )
    else:
        lines.append(
            "  No clear evidence the model learned a generalizable MD5 inverse."
            " This matches MD5's designed diffusion / one-way properties."
        )
    lines += [
        "",
        "Caveat: this studies learnability under a tiny enumerable plaintext",
        "space. It does not claim cryptanalytic breaks of MD5 at large.",
    ]
    return "\n".join(lines) + "\n"
