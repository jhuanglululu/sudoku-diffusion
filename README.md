# sudoku-diffusion

A tiny masked discrete diffusion transformer trained from scratch to solve 4×4
Sudoku. Vocabulary is 5 tokens (MASK, 1–4) over a 16-cell sequence. Training is
two-stage: SFT (denoising corrupted boards, including wrong digits so the model
learns to *remask* its own errors, plus a geometric-symmetry consistency loss)
then GRPO (group-relative RL on sampler trajectories, rewarding solved boards
and fewer sampler steps). The sampler can remask filled cells even after the
board is full, and boards can be solved from a completely empty start.

## Setup

```
uv sync
uv run scripts/gen_data.py   # writes datasets/eval.jsonl (fixed seed; committed)
```

No env vars. Development/tests run anywhere (CPU is fine); real runs on a CUDA
box (`get_device()` auto-picks the freest GPU). No checkpoint resume — runs
take minutes.

## Usage

```
# smoke check (~seconds on CPU)
uv run scripts/train.py --model tiny --training smoke --seed 0
uv run scripts/train.py --model tiny --training grpo-smoke --seed 0

# real runs
uv run scripts/train.py --model base --training sft --seed 0
uv run scripts/train.py --model base --training grpo --seed 0   # loads the sft checkpoint for the same model+seed

# solve-rate eval on the fixed eval set
uv run scripts/eval.py --model base --training sft --seed 0

# watch a single diffusion/remasking trajectory (--clues 0 = empty board)
uv run scripts/demo.py --model base --training sft --seed 0 --clues 5

# tests
uv run pytest
```

## Variations

**Model**
- tiny — smoke runs and shape checks, never real results
- base — the main experiment scale (<1M params)

**Training**
- smoke — 50-step local SFT sanity check
- sft — the standard denoising + consistency recipe
- grpo-smoke — 10-step local GRPO sanity check (starts from `smoke` checkpoint)
- grpo — GRPO from the `sft` checkpoint; rewards solving in fewer sampler steps

## Layout

- `records/<model>/<training>/<seed>/record.jsonl` — per-run metrics (meta line, then step/eval lines)
- `checkpoints/<model>/<training>/<seed>/current.safetensors` (+ `.json` sidecar) — latest weights
- `datasets/eval.jsonl` — fixed eval puzzles (0/4/5/6/7/8 clues), built from solution
  grids whose whole symmetry orbits are held out of training
