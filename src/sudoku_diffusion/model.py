"""Encoder-only transformer over 16 cells, logits over 5 tokens per cell."""

from __future__ import annotations

import subprocess

import torch
import torch.nn as nn

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


class SudokuDenoiser(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
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
        pos = torch.arange(self.cfg.seq_len, device=tokens.device)
        x = self.tok(tokens) + self.pos(pos)[None]
        return self.head(self.norm(self.encoder(x)))
