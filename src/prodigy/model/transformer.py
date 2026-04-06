"""
Decoder-only (GPT-style) Transformer for next-token drum prediction.

Architecture:
  token embedding  +  positional embedding
        ↓
  N × TransformerBlock
    ├─ LayerNorm
    ├─ CausalSelfAttention  (can't see future tokens — enforced by the mask)
    ├─ LayerNorm
    └─ Feed-Forward (Linear → GELU → Linear)
        ↓
  Final LayerNorm  →  Linear head  →  logits over vocabulary

Key design choices:
  - Pre-norm (LayerNorm before each sub-layer): more stable than post-norm.
  - Weight tying: the output projection shares weights with the token embedding.
    This reduces parameters and acts as a regulariser.
  - Learned positional embeddings (simpler than sinusoidal; works well in practice).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from prodigy.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a causal (autoregressive) mask.

    The mask sets all attention weights to -inf for positions j > i, so token i
    can only attend to tokens 0 … i. After softmax those entries become 0,
    meaning no information flows from the future.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # Fused Q, K, V projection — one matrix multiply instead of three
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)

        # Upper-triangular mask (True = masked out).  Registered as a buffer so
        # it moves to the right device automatically with model.to(device).
        causal_mask = torch.triu(torch.ones(config.seq_len, config.seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project and split into Q, K, V  →  each (B, T, C)
        q, k, v = self.qkv(x).split(C, dim=-1)

        # Reshape to (B, n_heads, T, head_dim) for batched attention
        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        # Scaled dot-product attention: softmax(Q Kᵀ / √dₖ) V
        scale = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) / scale          # (B, heads, T, T)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = (weights @ v)                                  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),                  # Smooth activation used in GPT-2/3
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections: x + f(x) let gradients flow directly to earlier layers
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class DrumTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.emb_drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)

        # Output head: maps each hidden state back to vocabulary logits
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share embedding and output projection weights.
        # Rationale: the embedding encodes "what does token X mean as input",
        # and the head encodes "how likely is token X as output" — they should
        # be similar representations.
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,              # (B, T) token ids
        targets: torch.Tensor | None = None,  # (B, T) for training
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.config.seq_len, f"Input length {T} > model seq_len {self.config.seq_len}"

        positions = torch.arange(T, device=idx.device)
        x = self.emb_drop(self.token_emb(idx) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x)

        logits = self.head(self.ln_f(x))   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten B×T into a single batch dimension for cross_entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
