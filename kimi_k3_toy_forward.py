#!/usr/bin/env python3
"""
kimi_k3_toy_forward.py
======================

A TINY, deliberately slow, heavily commented teaching script.

What this is
------------
A miniature model that implements the *ideas* behind Kimi K3's architecture:

  1. KDA-style linear / recurrent attention   (sequence mixing)
  2. Tiny "full" attention (stand-in for Gated MLA)
  3. Block Attention Residuals               (depth mixing)
  4. Latent MoE with SiTU-like activation    (channel / expert mixing)

What this is NOT
----------------
- Not the real Kimi K3 (2.8 trillion parameters).
- Not numerically matching Moonshot's checkpoints or API output.
- Not fast, not production, not quantized, not multimodal.

Why it exists
-------------
So a programmer who knows Python but not ML can *see* the data flowing
through the same kinds of passes K3 uses, without GPUs, without Hugging Face,
and without downloading terabytes of weights.

Run:
    python3 kimi_k3_toy_forward.py

Only dependency: NumPy  (pip install numpy)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# 0. Tiny "random seed" so runs are reproducible while you tinker
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)


# ===========================================================================
# PART A — Vocabulary of AI (in programmer terms)
# ===========================================================================
#
# Token
#   A chunk of text the model treats as one atomic symbol. Real models use
#   subword pieces ("running" -> "run" + "ning"). We use whole words for clarity.
#
# Embedding
#   A learned lookup table: token_id -> vector of floats (the "hidden size").
#   After this lookup, the model only manipulates arrays of numbers.
#
# Hidden state
#   The running representation of the sequence: shape (seq_len, hidden_size).
#   Every layer reads it, transforms it, and writes a new version.
#
# Logits
#   Raw scores over the vocabulary for "what token comes next?"
#   Softmax turns them into probabilities; we then pick the best (greedy).
#
# Autoregressive generation
#   Predict one next token, append it, repeat. Same loop ChatGPT/Kimi use.
#


# ===========================================================================
# PART B — Tiny tokenizer (so we can see strings <-> ids)
# ===========================================================================

VOCAB = [
    "<pad>", "<bos>", "<eos>",
    "the", "cat", "sat", "on", "mat",
    "dog", "ran", "to", "park",
    "and", "then", "slept", ".",
]
# Reverse map: word -> integer id
STOI = {w: i for i, w in enumerate(VOCAB)}
ITOS = {i: w for w, i in STOI.items()}


def encode(text: str) -> list[int]:
    """'the cat sat' -> [3, 4, 5]  (plus we will add <bos> ourselves)."""
    return [STOI[w] for w in text.strip().split()]


def decode(ids: list[int]) -> str:
    return " ".join(ITOS[i] for i in ids)


# ===========================================================================
# PART C — Small linear algebra helpers
# ===========================================================================
#
# Almost everything in a Transformer is:
#   y = x @ W   (+ optional bias)
# i.e. a matrix multiply that mixes channels of a vector.
#


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """
    x:      (..., in_dim)
    weight: (in_dim, out_dim)   NOTE: we store W as (in, out) for readability
    """
    y = x @ weight
    if bias is not None:
        y = y + bias
    return y


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    RMSNorm: scale each vector so its root-mean-square is ~1, then multiply
    by a learned gain. Used everywhere in modern LLMs instead of LayerNorm.
    """
    # x: (..., d)
    ms = np.mean(x * x, axis=-1, keepdims=True)
    x_hat = x / np.sqrt(ms + eps)
    return x_hat * weight


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax: exp(x) / sum(exp(x))."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softcap(x: np.ndarray, beta: float) -> np.ndarray:
    """
    Smooth clamp used by K3's SiTU activation:
        beta * tanh(x / beta)
    Near 0 it looks linear; large values flatten toward +/- beta.
    """
    return beta * np.tanh(x / beta)


def situ_glu(gate: np.ndarray, up: np.ndarray, beta1: float = 4.0, beta2: float = 25.0) -> np.ndarray:
    """
    Toy SiTU-GLU (Sigmoid Tanh Unit GLU), inspired by Kimi K3.

    Classic SwiGLU does roughly:  swish(gate) * up
    SiTU soft-caps both sides so huge activations cannot explode — important
    when experts are extremely sparse / low-precision.
    """
    gated = softcap(gate, beta1) * sigmoid(gate)
    upped = softcap(up, beta2)
    return gated * upped


def init_linear(in_dim: int, out_dim: int, scale: float = 0.2) -> np.ndarray:
    """Small random matrix. Real models use careful init; we just need nonzero."""
    return rng.normal(0.0, scale / math.sqrt(in_dim), size=(in_dim, out_dim)).astype(np.float64)


# ===========================================================================
# PART D — The three K3-style mixers (toy versions)
# ===========================================================================


# ---------------------------------------------------------------------------
# D1. KDA-style attention (sequence mixing with a FIXED-SIZE memory)
# ---------------------------------------------------------------------------
#
# Standard attention remembers EVERY past token (KV cache grows with length).
# Linear / delta-style attention keeps one matrix-sized "memory state" and
# updates it each step. Memory size does NOT grow with sequence length.
#
# Real KDA is much more carefully engineered (chunk kernels, channel-wise
# decay, short convolutions, etc.). Below is the pedagogical core:
#
#   S_t = alpha * (S_{t-1} - beta * outer(k, S_{t-1}^T k)) + beta * outer(k, v)
#   o_t = S_t^T @ q
#
# Read this as:
#   1) forget a bit of the old memory (alpha)
#   2) erase the stale association for this key (delta rule)
#   3) write the new value for this key
#   4) read with the query
#

@dataclass
class ToyKDA:
    d_model: int
    d_head: int

    def __post_init__(self) -> None:
        d, h = self.d_model, self.d_head
        self.Wq = init_linear(d, h)
        self.Wk = init_linear(d, h)
        self.Wv = init_linear(d, h)
        self.Wo = init_linear(h, d)
        # How strongly to forget / write (in real KDA these are input-dependent)
        self.log_alpha = np.array(-0.5)   # alpha = exp(log_alpha) in (0,1)
        self.beta = 0.5

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (T, d_model)  — whole sequence at once (prefill style)
        returns: (T, d_model)
        """
        T, _ = x.shape
        q = linear(x, self.Wq)  # (T, h)
        k = linear(x, self.Wk)
        v = linear(x, self.Wv)

        # Normalize q/k a bit (real KDA uses L2-norm in the kernel)
        q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
        k = k / (np.linalg.norm(k, axis=-1, keepdims=True) + 1e-8)

        alpha = float(np.exp(self.log_alpha))
        beta = float(self.beta)

        # Recurrent state S is (h, h): a FIXED-SIZE memory of the past.
        S = np.zeros((self.d_head, self.d_head), dtype=np.float64)
        outs = []
        for t in range(T):
            kt = k[t]  # (h,)
            vt = v[t]
            qt = q[t]

            # What does memory currently predict for this key?
            #   pred = S.T @ kt   (shape h,)   — value associated with kt
            pred = S.T @ kt

            # Delta update with decay:
            #   shrink old memory, erase wrong association, write new one
            S = alpha * (S - beta * np.outer(kt, pred))
            S = S + beta * np.outer(kt, vt)

            # Read
            ot = S.T @ qt
            outs.append(ot)

        o = np.stack(outs, axis=0)          # (T, h)
        return linear(o, self.Wo)           # (T, d_model)


# ---------------------------------------------------------------------------
# D2. Tiny full attention (stand-in for Gated MLA)
# ---------------------------------------------------------------------------
#
# Real K3 uses DeepSeek-style MLA: compress KV into a small latent, cache that,
# optionally NoPE + an output gate. Here we use classic causal self-attention
# so you can see the O(T^2) "compare every token to every previous token" idea.
#

@dataclass
class ToyFullAttention:
    d_model: int
    d_head: int

    def __post_init__(self) -> None:
        d, h = self.d_model, self.d_head
        self.Wq = init_linear(d, h)
        self.Wk = init_linear(d, h)
        self.Wv = init_linear(d, h)
        self.Wo = init_linear(h, d)
        # Output gate (K3's "Gated MLA" idea: modulate what we keep)
        self.Wg = init_linear(d, h)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (T, d_model)
        Causal mask: position t may only look at positions <= t.
        """
        T, _ = x.shape
        q = linear(x, self.Wq)  # (T, h)
        k = linear(x, self.Wk)
        v = linear(x, self.Wv)

        # scores[t, s] = q_t · k_s / sqrt(h)
        scale = 1.0 / math.sqrt(self.d_head)
        scores = (q @ k.T) * scale  # (T, T)

        # Causal mask: forbid looking at the future
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)

        weights = softmax(scores, axis=-1)   # each row sums to 1
        attn = weights @ v                   # (T, h)

        # Gate: y = sigmoid(x Wg) ⊙ attn   (then project back)
        gate = sigmoid(linear(x, self.Wg))
        attn = attn * gate
        return linear(attn, self.Wo)


# ---------------------------------------------------------------------------
# D3. Latent MoE (channel mixing with sparse experts)
# ---------------------------------------------------------------------------
#
# Instead of one big feed-forward network, MoE has many "experts" and only
# activates a few per token. K3's LatentMoE first projects to a smaller width
# (cheaper communication / smaller expert weights), runs experts there, then
# projects back. Shared experts always run on the full width.
#

@dataclass
class ToyLatentMoE:
    d_model: int
    d_latent: int
    d_ff: int
    n_experts: int = 4
    top_k: int = 2
    n_shared: int = 1

    def __post_init__(self) -> None:
        d, ell, f = self.d_model, self.d_latent, self.d_ff
        # Router chooses experts from the FULL hidden state (like real LatentMoE)
        self.W_router = init_linear(d, self.n_experts, scale=0.5)
        # Expert bias for load balancing — in K3 this is Quantile Balancing.
        # We keep a fixed toy bias; training would update it.
        self.expert_bias = np.zeros(self.n_experts, dtype=np.float64)

        # Down / up projections around the latent expert path
        self.W_down = init_linear(d, ell)
        self.W_up = init_linear(ell, d)
        self.latent_norm_w = np.ones(ell)

        # Each routed expert: gate_proj + up_proj + down_proj inside latent space
        self.experts = []
        for _ in range(self.n_experts):
            self.experts.append({
                "W_gate": init_linear(ell, f),
                "W_up": init_linear(ell, f),
                "W_down": init_linear(f, ell),
            })

        # Shared expert(s) on full width
        self.shared = []
        for _ in range(self.n_shared):
            self.shared.append({
                "W_gate": init_linear(d, f),
                "W_up": init_linear(d, f),
                "W_down": init_linear(f, d),
            })

    def _ffn(self, x: np.ndarray, params: dict, latent: bool) -> np.ndarray:
        gate = linear(x, params["W_gate"])
        up = linear(x, params["W_up"])
        h = situ_glu(gate, up)
        return linear(h, params["W_down"])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (T, d_model)
        For each token: pick top_k experts, mix their outputs, add shared experts.
        """
        T, d = x.shape
        # Router scores (sigmoid like DeepSeek / K3), bias only for selection
        scores = sigmoid(linear(x, self.W_router))            # (T, E)
        scores_for_choice = scores + self.expert_bias         # bias does NOT enter weights

        # Top-k expert indices per token
        top_idx = np.argsort(-scores_for_choice, axis=-1)[:, : self.top_k]  # (T, k)

        # Latent projection for routed path
        z = linear(x, self.W_down)  # (T, ell)

        routed = np.zeros_like(z)
        for t in range(T):
            idx = top_idx[t]
            w = scores[t, idx]
            w = w / (w.sum() + 1e-8)  # renormalize mixture weights
            acc = np.zeros(self.d_latent)
            for weight, e in zip(w, idx):
                acc = acc + weight * self._ffn(z[t], self.experts[e], latent=True)
            routed[t] = acc

        routed = rms_norm(routed, self.latent_norm_w)
        y = linear(routed, self.W_up)  # back to d_model

        # Shared experts always fire on the original x
        for shared in self.shared:
            y = y + self._ffn(x, shared, latent=False)
        return y


# ---------------------------------------------------------------------------
# D4. Block Attention Residuals (depth mixing)
# ---------------------------------------------------------------------------
#
# Normal residual:          h = h + layer(h)
#   -> every past layer is baked into ONE vector with equal weight 1.
#
# AttnRes idea:             h = sum_i  alpha_i * past_i
#   where alpha = softmax over depth, using a learned pseudo-query.
#
# Block AttnRes (what K3 uses): keep block summaries, attend over those
# instead of every individual layer (cheaper).
#

def apply_attn_res(
    prefix_sum: np.ndarray,
    block_residuals: list[np.ndarray],
    proj: np.ndarray,
    norm_w: np.ndarray,
) -> np.ndarray:
    """
    prefix_sum:      (T, d)  — current block's running sum of layer outputs
    block_residuals: list of (T, d) completed block summaries (+ embedding)
    proj:            (d,)    — the learned "pseudo-query" direction
    norm_w:          (d,)    — RMSNorm gain

    Returns a new (T, d) hidden state for the next sublayer.
    """
    # Stack sources along a new "depth" axis: (N+1, T, d)
    sources = np.stack(list(block_residuals) + [prefix_sum], axis=0)
    # Keys = RMSNorm(values); scores = key · (norm_w * proj)
    # This matches the spirit of K3's fused score weight.
    T = prefix_sum.shape[0]
    d = prefix_sum.shape[1]
    # Normalize each source vector per token
    ms = np.mean(sources * sources, axis=-1, keepdims=True)
    keys = sources / np.sqrt(ms + 1e-5)
    score_dir = norm_w * proj  # (d,)
    # scores: (N+1, T)
    scores = np.sum(keys * score_dir.reshape(1, 1, d), axis=-1)
    probs = softmax(scores, axis=0)  # attention over depth
    # Weighted sum of values
    out = np.sum(probs.reshape(-1, T, 1) * sources, axis=0)
    return out


# ===========================================================================
# PART E — One decoder layer and the full tiny model
# ===========================================================================

@dataclass
class ToyDecoderLayer:
    attn: object          # ToyKDA or ToyFullAttention
    moe: ToyLatentMoE
    d_model: int
    # AttnRes parameters (one for before-attn, one for before-mlp)
    attn_proj: np.ndarray
    attn_norm_w: np.ndarray
    mlp_proj: np.ndarray
    mlp_norm_w: np.ndarray
    input_norm_w: np.ndarray
    post_attn_norm_w: np.ndarray

    @classmethod
    def create(cls, d_model: int, d_head: int, d_latent: int, d_ff: int, use_kda: bool) -> "ToyDecoderLayer":
        attn = ToyKDA(d_model, d_head) if use_kda else ToyFullAttention(d_model, d_head)
        return cls(
            attn=attn,
            moe=ToyLatentMoE(d_model, d_latent, d_ff),
            d_model=d_model,
            # Zero-ish init for AttnRes queries ~ "start as uniform average"
            attn_proj=rng.normal(0, 0.01, size=(d_model,)),
            attn_norm_w=np.ones(d_model),
            mlp_proj=rng.normal(0, 0.01, size=(d_model,)),
            mlp_norm_w=np.ones(d_model),
            input_norm_w=np.ones(d_model),
            post_attn_norm_w=np.ones(d_model),
        )


@dataclass
class ToyKimiK3:
    """
    Hyper-small stack:

        layers = [KDA, KDA, FullAttn] * N_BLOCKS   # 2:1 toy of K3's 3:1
        block size for AttnRes = 3 sublayers' worth of (attn+mlp) pairs
                                 ≈ we treat each DecoderLayer as 2 'layers'
                                 and close a block every `attn_res_block` layers.

    Real K3: 93 layers, 3 KDA : 1 MLA, AttnRes block size 12, 896 experts...
    We use toy sizes so a laptop can run it in milliseconds.
    """
    vocab_size: int
    d_model: int
    n_layers: int
    attn_res_block: int  # in units of decoder layers

    def __post_init__(self) -> None:
        d = self.d_model
        self.embed = rng.normal(0, 0.2, size=(self.vocab_size, d))
        self.layers: list[ToyDecoderLayer] = []
        for i in range(self.n_layers):
            # Pattern: KDA, KDA, Full  (toy 2:1). Real K3 is KDA,KDA,KDA,MLA.
            use_kda = (i % 3) != 2
            self.layers.append(
                ToyDecoderLayer.create(
                    d_model=d,
                    d_head=max(8, d // 2),
                    d_latent=max(8, d // 2),
                    d_ff=d * 2,
                    use_kda=use_kda,
                )
            )
        self.final_norm_w = np.ones(d)
        self.lm_head = init_linear(d, self.vocab_size, scale=0.5)
        # Output AttnRes
        self.out_proj = rng.normal(0, 0.01, size=(d,))
        self.out_norm_w = np.ones(d)

    def forward(self, token_ids: list[int], verbose: bool = True) -> np.ndarray:
        """
        Full forward pass.

        Returns logits of shape (T, vocab_size).
        The LAST row is "scores for the next token after the whole prompt".
        """
        T = len(token_ids)
        # --- Pass 0: tokens -> vectors ---
        h = self.embed[np.array(token_ids)]  # (T, d)
        if verbose:
            print(f"\n[embed] tokens {token_ids} -> hidden shape {h.shape}")

        # block_residuals starts EMPTY; embedding becomes first block source
        # when we hit a block boundary (same spirit as K3 code).
        block_residuals: list[np.ndarray] = []
        prefix_sum = h.copy()  # running sum inside the current AttnRes block

        for i, layer in enumerate(self.layers):
            kind = "KDA" if isinstance(layer.attn, ToyKDA) else "FullAttn(MLA-stand-in)"
            if verbose:
                print(f"\n=== Decoder layer {i} [{kind}] ===")

            # --- AttnRes BEFORE attention ---
            if block_residuals:
                h_in = apply_attn_res(prefix_sum, block_residuals, layer.attn_proj, layer.attn_norm_w)
            else:
                # First block: nothing to attend over yet; use current prefix
                h_in = prefix_sum

            # At block boundaries, seal the current prefix into history
            if i % self.attn_res_block == 0:
                block_residuals.append(prefix_sum.copy())
                prefix_sum = None
                if verbose:
                    print(f"  AttnRes: sealed block #{len(block_residuals)} "
                          f"(now attending over {len(block_residuals)} block(s))")

            # --- Attention sublayer ---
            h_norm = rms_norm(h_in, layer.input_norm_w)
            attn_out = layer.attn.forward(h_norm)
            if prefix_sum is None:
                prefix_sum = attn_out
            else:
                prefix_sum = prefix_sum + attn_out
            if verbose:
                print(f"  attention out mean|abs|={np.mean(np.abs(attn_out)):.4f}")

            # --- AttnRes BEFORE MoE ---
            h_mlp_in = apply_attn_res(prefix_sum, block_residuals, layer.mlp_proj, layer.mlp_norm_w)

            # --- MoE / FFN sublayer ---
            h_norm2 = rms_norm(h_mlp_in, layer.post_attn_norm_w)
            moe_out = layer.moe.forward(h_norm2)
            prefix_sum = prefix_sum + moe_out
            if verbose:
                print(f"  moe out     mean|abs|={np.mean(np.abs(moe_out)):.4f}")
                print(f"  prefix_sum  mean|abs|={np.mean(np.abs(prefix_sum)):.4f}")

            h = prefix_sum  # carry forward

        # Final AttnRes + norm + vocabulary projection
        if block_residuals:
            h = apply_attn_res(h, block_residuals, self.out_proj, self.out_norm_w)
        h = rms_norm(h, self.final_norm_w)
        logits = linear(h, self.lm_head)  # (T, vocab)
        if verbose:
            print(f"\n[lm_head] logits shape {logits.shape}")
        return logits


# ===========================================================================
# PART F — Generation loop (the outer "chat" algorithm)
# ===========================================================================

def greedy_next_token(logits_row: np.ndarray, ban: set[int] | None = None) -> int:
    """Pick the vocabulary index with the highest score. No sampling."""
    row = logits_row.copy()
    if ban:
        for tid in ban:
            row[tid] = -1e9
    return int(np.argmax(row))


def generate(model: ToyKimiK3, prompt: str, max_new_tokens: int = 6) -> str:
    """
    Autoregressive decode:
      while not done:
          logits = model(tokens)          # full recompute each time (slow, clear)
          next = argmax(logits[-1])
          append next
    Real engines cache KDA state / MLA KV so they don't recompute the past.
    We intentionally recompute everything for clarity.
    """
    ids = [STOI["<bos>"]] + encode(prompt)
    print("=" * 72)
    print("PROMPT:", prompt)
    print("TOKEN IDS:", ids, "->", decode(ids))
    print("=" * 72)

    for step in range(max_new_tokens):
        print(f"\n########## GENERATE STEP {step} ##########")
        logits = model.forward(ids, verbose=True)
        # Next-token distribution from the last position.
        # Ban special tokens so the toy demo runs several visible steps;
        # real models learn when to emit <eos> themselves.
        last = logits[-1]
        ban = {STOI["<pad>"], STOI["<bos>"], STOI["<eos>"]}
        probs = softmax(last)
        nxt = greedy_next_token(last, ban=ban)

        # Show top-5 for intuition (among non-banned tokens)
        ranked = [t for t in np.argsort(-probs) if t not in ban][:5]
        print("\nTop-5 next-token candidates:")
        for tid in ranked:
            print(f"  {probs[tid]:6.3f}  {ITOS[tid]!r}")

        ids.append(nxt)
        print(f"\n>> Chose {nxt} ({ITOS[nxt]!r})")
        print(">> Text so far:", decode(ids))

        if nxt == STOI["."]:
            break

    return decode(ids)


# ===========================================================================
# PART G — main
# ===========================================================================

def main() -> None:
    print(
        """
This script is a MICROSCOPE, not a product.

Watch for these passes each generate step:
  1) embed tokens
  2) for each layer:
       AttnRes-read  ->  KDA or Full attention  -> accumulate
       AttnRes-read  ->  Latent MoE             -> accumulate
  3) final AttnRes + norm
  4) lm_head -> logits -> pick next token
  5) append token and repeat

Random weights => gibberish sentences. That is expected.
You are here to see STRUCTURE, not English quality.
"""
    )

    model = ToyKimiK3(
        vocab_size=len(VOCAB),
        d_model=32,       # real K3: 7168
        n_layers=6,       # real K3: 93
        attn_res_block=2, # real K3: 12 (in a related counting of sublayers)
    )

    # Count parameters the naive way (educational, not exact HF semantics)
    def nbytes(obj) -> int:
        total = 0
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        if isinstance(obj, dict):
            return sum(nbytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(nbytes(v) for v in obj)
        if hasattr(obj, "__dict__"):
            return sum(nbytes(v) for v in vars(obj).values())
        return 0

    approx_params = nbytes(model) // 8  # float64 bytes -> count
    print(f"Approx toy parameter tensors footprint: ~{approx_params:,} float64 values")
    print("(Real Kimi K3: ~2.8e12 parameters — about 100 million times larger.)\n")

    output = generate(model, "the cat sat on", max_new_tokens=5)
    print("\n" + "=" * 72)
    print("FINAL:", output)
    print("=" * 72)
    print(
        """
Suggested experiments (edit this file and re-run):
  - Set n_layers=1 and watch a single layer.
  - Force every layer to FullAttn (use_kda=False) and compare prints.
  - Print layer.moe top_idx to see which experts fired.
  - Replace greedy argmax with sampling: rng.choice(V, p=probs).
"""
    )


if __name__ == "__main__":
    main()
