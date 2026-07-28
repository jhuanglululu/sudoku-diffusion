"""Named model and training variations. New experiment = new named entry."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    vocab_size: int = 5
    seq_len: int = 16
    dropout: float = 0.0


class BaseTrainingConfig(BaseModel):
    name: str
    steps: int
    lr: float
    warmup_steps: int = 50
    log_every: int = 10
    val_every: int = 50
    # sampler (used by eval/demo for both kinds)
    max_sample_steps: int = 12
    commit_frac: float = 0.35
    # when set, confidence-rule committing replaces top-k (commit_frac ignored):
    # every masked cell whose max digit probability reaches it commits
    commit_threshold: float | None = None


class SFTConfig(BaseTrainingConfig):
    kind: Literal["sft"] = "sft"
    batch_size: int
    # corruption
    mask_frac_range: tuple[float, float] = (0.0, 1.0)  # 1.0 => fully empty boards seen in SFT
    wrong_frac_range: tuple[float, float] = (0.0, 0.3)
    consistency_weight: float = 1.0


class GRPOConfig(BaseTrainingConfig):
    kind: Literal["grpo"] = "grpo"
    init_from_training: str  # sft checkpoint to start from
    group_size: int = 8
    puzzles_per_batch: int = 16
    temperature: float = 1.0
    clip_eps: float = 0.2
    solved_bonus: float = 1.0
    efficiency_alpha: float = 0.5
    # 0 = start from an empty board; 1..3 have multiple solutions on 4x4,
    # fine since the reward verifies the board instead of matching one target
    clue_counts: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)


TrainingConfig = SFTConfig | GRPOConfig

MODELS: dict[str, ModelConfig] = {
    "tiny": ModelConfig(name="tiny", d_model=32, n_layers=2, n_heads=2, d_ff=64),
    "base": ModelConfig(name="base", d_model=128, n_layers=6, n_heads=4, d_ff=512),
}

TRAININGS: dict[str, TrainingConfig] = {
    "smoke": SFTConfig(name="smoke", steps=50, batch_size=32, lr=3e-4, warmup_steps=5, val_every=25),
    "sft": SFTConfig(name="sft", steps=4000, batch_size=256, lr=1e-3, warmup_steps=100, val_every=200),
    "grpo-smoke": GRPOConfig(
        name="grpo-smoke", steps=10, lr=1e-5, warmup_steps=0,
        init_from_training="smoke", group_size=4, puzzles_per_batch=4, log_every=1, val_every=5,
    ),
    "grpo-topk": GRPOConfig(
        name="grpo-topk", steps=500, lr=2e-5, warmup_steps=10,
        init_from_training="sft", group_size=8, puzzles_per_batch=32, log_every=5, val_every=25,
    ),
    "grpo-confidence": GRPOConfig(
        name="grpo-confidence", steps=500, lr=2e-5, warmup_steps=10,
        init_from_training="sft", group_size=8, puzzles_per_batch=32, log_every=5, val_every=25,
        commit_threshold=0.9,
    ),
}
