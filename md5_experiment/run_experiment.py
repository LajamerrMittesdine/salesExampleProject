#!/usr/bin/env python3
"""Run the MD5 learnability experiment end-to-end.

Example:
  python -m md5_experiment.run_experiment --epochs 40
  python -m md5_experiment.run_experiment --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python md5_experiment/run_experiment.py` from repo root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from md5_experiment.analyze import plot_history, save_json, scientific_verdict
from md5_experiment.config import ExperimentConfig
from md5_experiment.data import build_experiment_data
from md5_experiment.evaluate import (
    avalanche_probe,
    evaluate_forward,
    evaluate_inverse,
    memorization_probe,
    search_acceleration_probe,
)
from md5_experiment.train import train_forward, train_inverse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alphabet", default="0123456789")
    p.add_argument("--length", type=int, default=4, help="plaintext length")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("md5_experiment/artifacts"))
    p.add_argument("--no-forward", action="store_true")
    p.add_argument("--no-inverse", action="store_true")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Tiny smoke run (length=3 digits, few epochs)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ExperimentConfig(
        alphabet=args.alphabet,
        plaintext_length=args.length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        output_dir=args.output_dir,
        run_forward=not args.no_forward,
        run_inverse=not args.no_inverse,
    )
    if args.quick:
        cfg.plaintext_length = 3
        cfg.epochs = 15
        cfg.hidden_dims = (512, 512)
        cfg.batch_size = 128
        cfg.learning_rate = 2e-3

    cfg.validate()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config.json", cfg.to_dict())

    print(f"Device          : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Alphabet        : {cfg.alphabet!r}")
    print(f"Plaintext length: {cfg.plaintext_length}")
    print(f"Space size      : {cfg.space_size}")
    print(f"Epochs          : {cfg.epochs}")
    print(f"Output          : {out.resolve()}")

    data = build_experiment_data(cfg)
    print(
        f"Split sizes     : train={len(data.train.plaintexts)} "
        f"val={len(data.val.plaintexts)} test={len(data.test.plaintexts)}"
    )

    report: dict = {"config": cfg.to_dict()}

    print("\n=== Memorization probe (tiny subset fit, held-out transfer) ===")
    report["memorization"] = memorization_probe(cfg, data)
    print(
        f"Subset exact={report['memorization']['memorized_subset_exact_match']:.4f} "
        f"held-out exact={report['memorization']['held_out_exact_match']:.6f}"
    )

    if cfg.run_forward:
        print("\n=== Training forward model (plain → MD5 bits) ===")
        fwd_model, fwd_result = train_forward(cfg, data)
        torch.save(fwd_result.best_state_dict, out / "forward_model.pt")
        save_json(out / "forward_history.json", fwd_result.history)
        plot_history(
            fwd_result.history,
            [
                ("train_bit_accuracy", "Train bit accuracy"),
                ("val_bit_accuracy", "Val bit accuracy"),
                ("train_exact_match", "Train exact match"),
                ("val_exact_match", "Val exact match"),
            ],
            out / "forward_history.png",
            "Forward model learning curves",
        )
        report["forward"] = evaluate_forward(fwd_model, cfg, data)
        report["avalanche"] = avalanche_probe(fwd_model, cfg)
        print(
            f"Forward test bit_acc={report['forward']['test']['bit_accuracy']:.4f} "
            f"exact={report['forward']['test']['exact_match']:.6f}"
        )
        print(
            f"Avalanche true={report['avalanche']['true_mean_bit_flip_fraction']:.3f} "
            f"pred={report['avalanche']['pred_mean_bit_flip_fraction']:.3f}"
        )

    if cfg.run_inverse:
        print("\n=== Training inverse model (MD5 bits → plain) ===")
        inv_model, inv_result = train_inverse(cfg, data)
        torch.save(inv_result.best_state_dict, out / "inverse_model.pt")
        save_json(out / "inverse_history.json", inv_result.history)
        plot_history(
            inv_result.history,
            [
                ("train_char_accuracy", "Train char accuracy"),
                ("val_char_accuracy", "Val char accuracy"),
                ("train_exact_match", "Train exact match"),
                ("val_exact_match", "Val exact match"),
            ],
            out / "inverse_history.png",
            "Inverse model learning curves",
        )
        report["inverse"] = evaluate_inverse(inv_model, cfg, data)
        report["search"] = search_acceleration_probe(inv_model, cfg, data)
        print(
            f"Inverse train exact={report['inverse']['train']['exact_match']:.4f} "
            f"test exact={report['inverse']['test']['exact_match']:.6f}"
        )
        print(
            f"Search mean_rank={report['search']['mean_rank']:.1f} "
            f"(random≈{report['search']['expected_random_mean_rank']:.1f})"
        )

    save_json(out / "report.json", report)
    verdict = scientific_verdict(report)
    (out / "summary.txt").write_text(verdict, encoding="utf-8")
    print("\n" + verdict)
    print(f"Wrote artifacts under {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
