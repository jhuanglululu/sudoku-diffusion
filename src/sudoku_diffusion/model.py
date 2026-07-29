"""Denoiser over 16 cells, logits over 5 tokens per cell.

Two architectures (ModelConfig.arch):
- "transformer": encoder-only transformer with learned positional embeddings.
- "unit": GNN-style message passing on the sudoku constraint graph. Each
  block: unit-masked attention (content-based selection over the 7 peers,
  with a learned bias per relation type) -> unit-masked mix (typed sums over
  row/col/box peers with per-channel scales — the counting path softmax
  attention lacks) -> channel MLP. No positional embeddings: every sublayer
  commutes with the geometric symmetry group, so the model is exactly
  equivariant; cell identity comes from clue content alone, and greedy
  sampling needs the sampler's stochastic warmup on symmetric boards.
"""

from __future__ import annotations

import subprocess

import torch
import torch.nn as nn

from .data import BOXES, CELLS, COLS, ROWS
from .variations import ModelConfig


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{_free_gpu_index()}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _free_gpu_index() -> int:
    """Pick the GPU with the most free memory (shared training boxes)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.split()
        return max(range(len(out)), key=lambda i: int(out[i]))
    except Exception:
        return 0


def _mask_from_units(units: list[tuple[int, ...]]) -> torch.Tensor:
    """(16, 16) bool: m[i, j] = i and j share a unit (self excluded)."""
    m = torch.zeros(CELLS, CELLS, dtype=torch.bool)
    for u in units:
        for i in u:
            for j in u:
                if i != j:
                    m[i, j] = True
    return m


class UnitAttention(nn.Module):
    """Multi-head attention restricted to the 7 same-unit peers, with a
    learned per-head additive bias for each relation type (row/col/box)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.rel = nn.Parameter(torch.zeros(3, cfg.n_heads))  # row/col/box bias
        masks = torch.stack([_mask_from_units(u) for u in (ROWS, COLS, BOXES)])
        self.register_buffer("rel_masks", masks, persistent=False)          # (3, 16, 16)
        self.register_buffer("peers", masks.any(0), persistent=False)      # (16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        dh = D // self.n_heads
        q, k, v = self.qkv(x).view(B, S, 3, self.n_heads, dh).permute(2, 0, 3, 1, 4)
        scores = (q @ k.transpose(-2, -1)) * dh**-0.5                       # (B, H, 16, 16)
        bias = (self.rel[:, :, None, None] * self.rel_masks[:, None]).sum(0)  # (H, 16, 16)
        scores = (scores + bias).masked_fill(~self.peers, float("-inf"))
        out = torch.softmax(scores, dim=-1) @ v                             # (B, H, 16, dh)
        return self.drop(self.proj(out.transpose(1, 2).reshape(B, S, D)))


class UnitMix(nn.Module):
    """Typed sums over row/col/box peers, scaled per channel. Sums (not
    softmax) so digit counts per unit are directly expressible; position-
    independent scales keep the sublayer equivariant."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(3, cfg.d_model))  # starts as a no-op
        masks = torch.stack([_mask_from_units(u) for u in (ROWS, COLS, BOXES)])
        self.register_buffer("rel_masks", masks.float(), persistent=False)  # (3, 16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        agg = torch.einsum("tij,bjd->tbid", self.rel_masks, x)  # (3, B, 16, D)
        return (self.scale[:, None, None] * agg).sum(0)


class UnitBlock(nn.Module):
    """Pre-norm residual block: select (attention) -> aggregate (mix) ->
    compute (channel MLP)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = UnitAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mix = UnitMix(cfg)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mix(self.norm2(x))
        return x + self.ffn(self.norm3(x))


class SudokuDenoiser(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.arch == "unit":
            self.blocks = nn.ModuleList(UnitBlock(cfg) for _ in range(cfg.n_layers))
        else:
            self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
                dropout=cfg.dropout, batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, 16) long -> logits (B, 16, 5)."""
        if self.cfg.arch == "unit":
            x = self.tok(tokens)
            for block in self.blocks:
                x = block(x)
        else:
            pos = torch.arange(self.cfg.seq_len, device=tokens.device)
            x = self.tok(tokens) + self.pos(pos)[None]
            x = self.encoder(x)
        return self.head(self.norm(x))
