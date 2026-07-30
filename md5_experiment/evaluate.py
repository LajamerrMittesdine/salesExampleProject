"""Held-out evaluation and search-acceleration probes."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import ExperimentData, ForwardDataset, InverseDataset, encode_plaintext_onehot
from .train import forward_metrics, inverse_metrics


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_forward(
    model: nn.Module, cfg: ExperimentConfig, data: ExperimentData
) -> dict[str, Any]:
    device = _device()
    model = model.to(device)
    out: dict[str, Any] = {}
    for name, split in (("train", data.train), ("val", data.val), ("test", data.test)):
        loader = DataLoader(
            ForwardDataset(split), batch_size=cfg.batch_size, shuffle=False
        )
        out[name] = forward_metrics(model, loader, device)

    # Chance baselines for interpreting bit accuracy.
    out["baselines"] = {
        "random_bit_accuracy": 0.5,
        "note": (
            "MD5 bits on constrained plaintexts are near-balanced; "
            "~0.5 bit accuracy is chance. Exact digest match by chance is ~2^-128."
        ),
    }
    return out


def evaluate_inverse(
    model: nn.Module, cfg: ExperimentConfig, data: ExperimentData
) -> dict[str, Any]:
    device = _device()
    model = model.to(device)
    out: dict[str, Any] = {}
    for name, split in (("train", data.train), ("val", data.val), ("test", data.test)):
        loader = DataLoader(
            InverseDataset(split), batch_size=cfg.batch_size, shuffle=False
        )
        out[name] = inverse_metrics(
            model, loader, device, cfg.plaintext_length, cfg.alphabet_size
        )

    out["baselines"] = {
        "random_char_accuracy": 1.0 / cfg.alphabet_size,
        "random_exact_match": 1.0 / cfg.space_size,
        "note": (
            "Generalization means test exact_match >> random_exact_match. "
            "High train / near-chance test implies memorization, not learning MD5."
        ),
    }
    return out


@torch.no_grad()
def search_acceleration_probe(
    model: nn.Module,
    cfg: ExperimentConfig,
    data: ExperimentData,
    max_targets: int = 200,
) -> dict[str, Any]:
    """Rank all training+val+test candidates by model score for held-out hashes.

    If the inverse model learned structure useful for preimage search, true
    plaintexts should rank near the top among the full enumerable space.
    If ranks are uniform, the model provides no search acceleration.
    """
    device = _device()
    model = model.to(device).eval()

    # Candidate pool = entire enumerable space represented in our splits.
    all_plain = data.train.plaintexts + data.val.plaintexts + data.test.plaintexts
    all_chars = np.concatenate(
        [data.train.y_chars, data.val.y_chars, data.test.y_chars], axis=0
    )
    candidate_labels = torch.from_numpy(all_chars).to(device)  # (N, L)

    targets = data.test.plaintexts[:max_targets]
    target_bits = torch.from_numpy(data.test.y_bits[:max_targets]).to(device)

    ranks: list[int] = []
    top1 = 0
    top10 = 0
    length = cfg.plaintext_length
    alpha = cfg.alphabet_size

    # Score all candidates once per target via log-prob of predicted chars.
    # For each target hash, compute log-softmax over alphabet per position,
    # then score every candidate string.
    for i in range(target_bits.shape[0]):
        logits = model(target_bits[i : i + 1]).view(length, alpha)
        log_probs = torch.log_softmax(logits, dim=-1)  # (L, A)
        # Score each candidate string: sum over positions of log p(char).
        scores = torch.zeros(candidate_labels.shape[0], device=device)
        for pos in range(length):
            scores += log_probs[pos][candidate_labels[:, pos]]

        # True plaintext index among all_plain
        true_idx = all_plain.index(targets[i])
        order = torch.argsort(scores, descending=True)
        rank = int((order == true_idx).nonzero(as_tuple=False).item()) + 1
        ranks.append(rank)
        if rank == 1:
            top1 += 1
        if rank <= 10:
            top10 += 1

    ranks_arr = np.array(ranks, dtype=np.float64)
    n_space = len(all_plain)
    return {
        "n_targets": len(ranks),
        "space_size": n_space,
        "mean_rank": float(ranks_arr.mean()),
        "median_rank": float(np.median(ranks_arr)),
        "expected_random_mean_rank": (n_space + 1) / 2.0,
        "top1_hit_rate": top1 / max(len(ranks), 1),
        "top10_hit_rate": top10 / max(len(ranks), 1),
        "interpretation": (
            "Ranks near random mean ⇒ no useful learned preimage prior. "
            "Mean rank << random ⇒ model accelerates search within this space."
        ),
    }


def memorization_probe(
    cfg: ExperimentConfig,
    data: ExperimentData,
    n_memorize: int = 64,
    epochs: int = 200,
) -> dict[str, Any]:
    """Show that memorizing a tiny subset is easy, but does not transfer.

    This separates "can nets fit MD5 pairs at all?" from "do they learn MD5?"
    """
    from .models import build_inverse_model

    device = _device()
    n = min(n_memorize, len(data.train.plaintexts))
    x = torch.from_numpy(data.train.y_bits[:n]).to(device)
    y = torch.from_numpy(data.train.y_chars[:n]).to(device)
    model = build_inverse_model(
        cfg.hash_bits,
        cfg.plaintext_length,
        cfg.alphabet_size,
        hidden_dims=(512, 512, 512),
        dropout=0.0,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    ce = nn.CrossEntropyLoss()
    length, alpha = cfg.plaintext_length, cfg.alphabet_size

    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(x).view(-1, length, alpha)
        loss = ce(logits.reshape(-1, alpha), y.reshape(-1))
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        mem_pred = model(x).view(-1, length, alpha).argmax(dim=-1)
        mem_exact = float((mem_pred == y).all(dim=1).float().mean().item())

        x_te = torch.from_numpy(data.test.y_bits).to(device)
        y_te = torch.from_numpy(data.test.y_chars).to(device)
        te_pred = model(x_te).view(-1, length, alpha).argmax(dim=-1)
        te_exact = float((te_pred == y_te).all(dim=1).float().mean().item())
        te_char = float((te_pred == y_te).float().mean().item())

    return {
        "n_memorize": n,
        "epochs": epochs,
        "memorized_subset_exact_match": mem_exact,
        "held_out_exact_match": te_exact,
        "held_out_char_accuracy": te_char,
        "random_exact_match": 1.0 / cfg.space_size,
        "interpretation": (
            "High subset exact_match + near-chance held-out means the architecture "
            "can store MD5 pairs as a lookup table, but that storage does not "
            "reveal a general inverse of MD5."
        ),
    }


@torch.no_grad()
def avalanche_probe(
    model: nn.Module, cfg: ExperimentConfig, n_samples: int = 200
) -> dict[str, Any]:
    """Compare true MD5 avalanche to the forward model's predicted avalanche.

    A model that internalized MD5 should flip ~50% of output bits when a single
    input symbol changes. A memorizer / weak learner often under-avalanches.
    """
    from .data import md5_bits

    device = _device()
    model = model.to(device).eval()
    rng = np.random.default_rng(cfg.seed + 7)
    alphabet = cfg.alphabet

    true_flip_fracs: list[float] = []
    pred_flip_fracs: list[float] = []

    for _ in range(n_samples):
        chars = [alphabet[int(rng.integers(0, len(alphabet)))] for _ in range(cfg.plaintext_length)]
        pos = int(rng.integers(0, cfg.plaintext_length))
        alt = alphabet[int(rng.integers(0, len(alphabet)))]
        while alt == chars[pos]:
            alt = alphabet[int(rng.integers(0, len(alphabet)))]
        other = chars.copy()
        other[pos] = alt
        a, b = "".join(chars), "".join(other)

        ta = md5_bits(a)
        tb = md5_bits(b)
        true_flip_fracs.append(float(np.mean(ta != tb)))

        xa = torch.from_numpy(encode_plaintext_onehot(a, alphabet)).unsqueeze(0).to(device)
        xb = torch.from_numpy(encode_plaintext_onehot(b, alphabet)).unsqueeze(0).to(device)
        pa = (model(xa) >= 0.5).float().cpu().numpy()[0]
        pb = (model(xb) >= 0.5).float().cpu().numpy()[0]
        pred_flip_fracs.append(float(np.mean(pa != pb)))

    return {
        "n_samples": n_samples,
        "true_mean_bit_flip_fraction": float(np.mean(true_flip_fracs)),
        "pred_mean_bit_flip_fraction": float(np.mean(pred_flip_fracs)),
        "true_std": float(np.std(true_flip_fracs)),
        "pred_std": float(np.std(pred_flip_fracs)),
        "ideal_avalanche": 0.5,
        "interpretation": (
            "MD5's avalanche is ~0.5. If the model predicts far less, it has not "
            "learned the diffusion structure of MD5."
        ),
    }
