"""Run bookkeeping: seeding, records JSONL, checkpoints."""

from __future__ import annotations

import json
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_dir(base: str, model: str, training: str, seed: int) -> Path:
    d = ROOT / base / model / training / str(seed)
    d.mkdir(parents=True, exist_ok=True)
    return d


class Record:
    def __init__(self, model: str, training: str, seed: int, config: dict):
        self.path = run_dir("records", model, training, seed) / "record.jsonl"
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip() or None
        self.write(
            type="meta", model=model, training=training, seed=seed, config=config,
            git_commit=commit, started=datetime.now().isoformat(timespec="seconds"),
        )

    def write(self, **fields) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(fields) + "\n")


def save_checkpoint(model_nn: torch.nn.Module, model: str, training: str, seed: int, step: int, extra: dict) -> None:
    d = run_dir("checkpoints", model, training, seed)
    state = {k: v.contiguous().cpu() for k, v in model_nn.state_dict().items()}
    save_file(state, str(d / "current.safetensors"))
    (d / "current.json").write_text(json.dumps({"step": step, **extra}))


def load_checkpoint(model_nn: torch.nn.Module, model: str, training: str, seed: int) -> dict:
    d = run_dir("checkpoints", model, training, seed)
    model_nn.load_state_dict(load_file(str(d / "current.safetensors")))
    return json.loads((d / "current.json").read_text())
