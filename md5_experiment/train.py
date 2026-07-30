"""Training loops for forward and inverse MD5 models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import ExperimentData, ForwardDataset, InverseDataset
from .models import build_forward_model, build_inverse_model


@dataclass
class TrainResult:
    history: list[dict]
    best_state_dict: dict
    best_val_metric: float
    metric_name: str


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def forward_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_bits = 0
    correct_bits = 0
    exact = 0
    n = 0
    bce = nn.BCELoss(reduction="sum")
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss_sum += float(bce(pred, y).item())
        bits = (pred >= 0.5).float()
        correct_bits += int((bits == y).sum().item())
        total_bits += y.numel()
        exact += int((bits == y).all(dim=1).sum().item())
        n += y.shape[0]
    return {
        "loss": loss_sum / max(total_bits, 1),
        "bit_accuracy": correct_bits / max(total_bits, 1),
        "exact_match": exact / max(n, 1),
    }


@torch.no_grad()
def inverse_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    length: int,
    alphabet_size: int,
) -> dict:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    char_correct = 0
    char_total = 0
    exact = 0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).view(-1, length, alphabet_size)
        loss_sum += float(
            ce(logits.reshape(-1, alphabet_size), y.reshape(-1)).item()
        )
        pred = logits.argmax(dim=-1)
        char_correct += int((pred == y).sum().item())
        char_total += y.numel()
        exact += int((pred == y).all(dim=1).sum().item())
        n += y.shape[0]
    return {
        "loss": loss_sum / max(char_total, 1),
        "char_accuracy": char_correct / max(char_total, 1),
        "exact_match": exact / max(n, 1),
    }


def _train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: ExperimentConfig,
    loss_fn: Callable,
    eval_fn: Callable,
    maximize_metric: str,
) -> TrainResult:
    device = _device()
    model = model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    history: list[dict] = []
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = -1.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model, x, y)
            loss.backward()
            opt.step()
            running += float(loss.item())
            steps += 1

        train_m = eval_fn(model, train_loader, device)
        val_m = eval_fn(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss_step": running / max(steps, 1),
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        history.append(row)
        score = float(val_m[maximize_metric])
        if score > best_val:
            best_val = score
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    return TrainResult(
        history=history,
        best_state_dict=best_state,
        best_val_metric=best_val,
        metric_name=f"val_{maximize_metric}",
    )


def train_forward(cfg: ExperimentConfig, data: ExperimentData) -> tuple[nn.Module, TrainResult]:
    model = build_forward_model(
        plaintext_bits=cfg.plaintext_bits,
        hash_bits=cfg.hash_bits,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    )
    train_loader = DataLoader(
        ForwardDataset(data.train), batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        ForwardDataset(data.val), batch_size=cfg.batch_size, shuffle=False
    )
    bce = nn.BCELoss()

    def loss_fn(m, x, y):
        return bce(m(x), y)

    def eval_fn(m, loader, device):
        return forward_metrics(m, loader, device)

    result = _train_loop(
        model,
        train_loader,
        val_loader,
        cfg,
        loss_fn,
        eval_fn,
        # Bit accuracy is smoother than exact digest match early in training.
        maximize_metric="bit_accuracy",
    )
    model.load_state_dict(result.best_state_dict)
    return model, result


def train_inverse(cfg: ExperimentConfig, data: ExperimentData) -> tuple[nn.Module, TrainResult]:
    model = build_inverse_model(
        hash_bits=cfg.hash_bits,
        plaintext_length=cfg.plaintext_length,
        alphabet_size=cfg.alphabet_size,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    )
    train_loader = DataLoader(
        InverseDataset(data.train), batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        InverseDataset(data.val), batch_size=cfg.batch_size, shuffle=False
    )
    ce = nn.CrossEntropyLoss()
    length = cfg.plaintext_length
    alpha = cfg.alphabet_size

    def loss_fn(m, x, y):
        logits = m(x).view(-1, length, alpha)
        return ce(logits.reshape(-1, alpha), y.reshape(-1))

    def eval_fn(m, loader, device):
        return inverse_metrics(m, loader, device, length, alpha)

    result = _train_loop(
        model,
        train_loader,
        val_loader,
        cfg,
        loss_fn,
        eval_fn,
        # Prefer char accuracy for checkpointing; exact-match is sparse early on.
        maximize_metric="char_accuracy",
    )
    model.load_state_dict(result.best_state_dict)
    return model, result
