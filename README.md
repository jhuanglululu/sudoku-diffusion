# sudoku-diffusion

A tiny masked discrete diffusion transformer trained from scratch to solve 4×4
Sudoku. Vocabulary is 5 tokens (MASK, 1–4) over a 16-cell sequence. Training is
two-stage: SFT (denoising corrupted boards, including wrong digits so the model
learns to *remask* its own errors, plus a geometric-symmetry consistency loss)
then GRPO (group-relative RL on sampler trajectories, rewarding solved boards
and fewer sampler steps). Each sampler step, every masked cell picks among all
5 tokens: a digit commits the cell, MASK means "not decided yet" — so the model
paces its own commits, and the GRPO step-efficiency reward trains that pacing.
The sampler can remask filled cells even after the board is full, and boards
can be solved from a completely empty start.

## Setup

```
uv sync
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

# greedy solve-rate eval on freshly generated eval-orbit puzzles (0-8 clues)
uv run scripts/eval.py --model base --training sft --seed 0 --puzzle-seed 0 --n 100

# watch a single diffusion/remasking trajectory (--clues 0 = empty board)
uv run scripts/demo.py --model base --training sft --seed 0 --clues 5

# tests
uv run pytest
```

## Variations

**Model**
- tiny — smoke runs and shape checks, never real results
- base — encoder transformer, the main experiment scale (~3.6M params)
- unit — GNN-style message passing on the constraint graph (unit-masked
  attention + typed peer sums + channel MLP), no positional embeddings,
  exactly equivariant under the symmetry group; same scale as base

**Training**
- smoke — 50-step local SFT sanity check
- sft — the standard denoising + consistency recipe
- grpo-smoke — 10-step local GRPO sanity check (starts from `smoke` checkpoint)
- grpo — GRPO from the `sft` checkpoint; rewards solving in fewer sampler steps

## Layout

- `records/<model>/<training>/<seed>/record.jsonl` — per-run metrics (meta line, then step/eval lines)
- `checkpoints/<model>/<training>/<seed>/current.safetensors` (+ `.json` sidecar) — latest weights;
  `best.safetensors` — best weights by val loss (SFT) / solve rate (GRPO). eval and demo
  load `current` (latest); GRPO init loads `best`, falling back to `current`

Eval puzzles are generated on the fly (`--puzzle-seed`, `--n` per clue count) from
solution grids whose whole symmetry orbits are held out of training.
