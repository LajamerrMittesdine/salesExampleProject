"""Dataset construction for constrained MD5 learning experiments."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ExperimentConfig


def md5_hex(plaintext: str) -> str:
    return hashlib.md5(plaintext.encode("ascii")).hexdigest()


def md5_bits(plaintext: str) -> np.ndarray:
    digest = hashlib.md5(plaintext.encode("ascii")).digest()
    bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))
    return bits.astype(np.float32)


def enumerate_plaintexts(alphabet: str, length: int) -> list[str]:
    return ["".join(chars) for chars in itertools.product(alphabet, repeat=length)]


def encode_plaintext_onehot(plaintext: str, alphabet: str) -> np.ndarray:
    index = {ch: i for i, ch in enumerate(alphabet)}
    vec = np.zeros(len(plaintext) * len(alphabet), dtype=np.float32)
    for pos, ch in enumerate(plaintext):
        vec[pos * len(alphabet) + index[ch]] = 1.0
    return vec


def encode_plaintext_indices(plaintext: str, alphabet: str) -> np.ndarray:
    index = {ch: i for i, ch in enumerate(alphabet)}
    return np.array([index[ch] for ch in plaintext], dtype=np.int64)


def hash_hex_to_bits(hex_digest: str) -> np.ndarray:
    raw = bytes.fromhex(hex_digest)
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).astype(np.float32)


@dataclass
class SplitArrays:
    plaintexts: list[str]
    hashes_hex: list[str]
    x_onehot: np.ndarray
    y_bits: np.ndarray
    y_chars: np.ndarray


@dataclass
class ExperimentData:
    train: SplitArrays
    val: SplitArrays
    test: SplitArrays
    alphabet: str
    plaintext_length: int


def _build_split(plaintexts: list[str], alphabet: str) -> SplitArrays:
    hashes_hex = [md5_hex(p) for p in plaintexts]
    x_onehot = np.stack([encode_plaintext_onehot(p, alphabet) for p in plaintexts])
    y_bits = np.stack([md5_bits(p) for p in plaintexts])
    y_chars = np.stack([encode_plaintext_indices(p, alphabet) for p in plaintexts])
    return SplitArrays(
        plaintexts=plaintexts,
        hashes_hex=hashes_hex,
        x_onehot=x_onehot,
        y_bits=y_bits,
        y_chars=y_chars,
    )


def build_experiment_data(cfg: ExperimentConfig) -> ExperimentData:
    cfg.validate()
    all_plain = enumerate_plaintexts(cfg.alphabet, cfg.plaintext_length)
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(all_plain))
    plain_shuffled = [all_plain[i] for i in order]

    n = len(plain_shuffled)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    train_p = plain_shuffled[:n_train]
    val_p = plain_shuffled[n_train : n_train + n_val]
    test_p = plain_shuffled[n_train + n_val :]

    return ExperimentData(
        train=_build_split(train_p, cfg.alphabet),
        val=_build_split(val_p, cfg.alphabet),
        test=_build_split(test_p, cfg.alphabet),
        alphabet=cfg.alphabet,
        plaintext_length=cfg.plaintext_length,
    )


class ForwardDataset(Dataset):
    """plain one-hot → MD5 bit vector."""

    def __init__(self, split: SplitArrays):
        self.x = torch.from_numpy(split.x_onehot)
        self.y = torch.from_numpy(split.y_bits)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class InverseDataset(Dataset):
    """MD5 bit vector → per-position character class indices."""

    def __init__(self, split: SplitArrays):
        self.x = torch.from_numpy(split.y_bits)
        self.y = torch.from_numpy(split.y_chars)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
