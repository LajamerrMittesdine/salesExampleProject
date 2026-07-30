"""Experiment configuration.

The input space is deliberately tiny and fully enumerable so we can measure
true generalization (held-out plaintexts) rather than "did we crawl the web
for preimage tables."
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    # Constrained plaintext space: fixed-length digit strings, e.g. "0042".
    alphabet: str = "0123456789"
    plaintext_length: int = 4

    # Data split (by plaintext identity — no leakage).
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42

    # Model / training
    # Generous capacity + light regularization: we want the net to be *able*
    # to memorize the training table so that held-out failure is informative.
    hidden_dims: tuple[int, ...] = (1024, 1024, 1024)
    dropout: float = 0.0
    batch_size: int = 256
    epochs: int = 80
    learning_rate: float = 2e-3
    weight_decay: float = 0.0

    # Tasks to run: forward (plain→hash bits), inverse (hash→plain chars)
    run_forward: bool = True
    run_inverse: bool = True

    # Output
    output_dir: Path = field(default_factory=lambda: Path("md5_experiment/artifacts"))

    def validate(self) -> None:
        if abs(self.train_frac + self.val_frac + self.test_frac - 1.0) > 1e-9:
            raise ValueError("train/val/test fractions must sum to 1")
        if self.plaintext_length < 1:
            raise ValueError("plaintext_length must be >= 1")
        if len(self.alphabet) < 2:
            raise ValueError("alphabet must have at least 2 symbols")

    @property
    def space_size(self) -> int:
        return len(self.alphabet) ** self.plaintext_length

    @property
    def alphabet_size(self) -> int:
        return len(self.alphabet)

    @property
    def hash_bits(self) -> int:
        return 128  # MD5 digest size

    @property
    def plaintext_bits(self) -> int:
        # One-hot over alphabet per position, flattened for the MLP input.
        return self.plaintext_length * self.alphabet_size

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        d["space_size"] = self.space_size
        return d
