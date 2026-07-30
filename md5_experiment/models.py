"""MLP models for forward (plain→hash) and inverse (hash→plain) tasks."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: tuple[int, ...] = (512, 512, 512),
        dropout: float = 0.1,
        out_activation: str | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.out_activation = out_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.out_activation == "sigmoid":
            return torch.sigmoid(y)
        return y


def build_forward_model(
    plaintext_bits: int,
    hash_bits: int = 128,
    hidden_dims: tuple[int, ...] = (512, 512, 512),
    dropout: float = 0.1,
) -> MLP:
    # Sigmoid outputs: independent bit probabilities for the 128-bit digest.
    return MLP(
        in_dim=plaintext_bits,
        out_dim=hash_bits,
        hidden_dims=hidden_dims,
        dropout=dropout,
        out_activation="sigmoid",
    )


def build_inverse_model(
    hash_bits: int,
    plaintext_length: int,
    alphabet_size: int,
    hidden_dims: tuple[int, ...] = (512, 512, 512),
    dropout: float = 0.1,
) -> MLP:
    # Logits shaped as (batch, length * alphabet); reshaped in the loss/metrics.
    return MLP(
        in_dim=hash_bits,
        out_dim=plaintext_length * alphabet_size,
        hidden_dims=hidden_dims,
        dropout=dropout,
        out_activation=None,
    )
