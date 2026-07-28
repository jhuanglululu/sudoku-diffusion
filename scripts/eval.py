"""Greedy solve-rate eval on freshly generated eval-orbit puzzles.

uv run scripts/eval.py --model tiny --training sft --seed 0 --puzzle-seed 0 --n 100
"""

import argparse
import json

import numpy as np
import torch

from sudoku_diffusion.data import SPLIT_SEED, orbit_split, random_puzzle, solved
from sudoku_diffusion.model import SudokuDenoiser, get_device
from sudoku_diffusion.runs import load_checkpoint, run_dir
from sudoku_diffusion.sampler import sample
from sudoku_diffusion.variations import MODELS, TRAININGS

CLUE_COUNTS = (0, 1, 2, 3, 4, 5, 6, 7, 8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--training", required=True, choices=sorted(TRAININGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--puzzle-seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=100, help="puzzles per clue count")
    args = ap.parse_args()
    cfg = TRAININGS[args.training]

    device = get_device()
    model = SudokuDenoiser(MODELS[args.model]).to(device)
    meta = load_checkpoint(model, args.model, args.training, args.seed)
    print(f"loaded {meta['checkpoint']} checkpoint (step {meta.get('step')})")
    model.eval()

    rng = np.random.default_rng(args.puzzle_seed)
    _, eval_sols = orbit_split(np.random.default_rng(SPLIT_SEED))
    entries: list[tuple[int, np.ndarray]] = []
    for n_clues in CLUE_COUNTS:
        if n_clues == 0:
            # greedy sampling is deterministic, one empty board is enough
            entries.append((0, np.zeros(16, dtype=eval_sols.dtype)))
            continue
        for _ in range(args.n):
            sol = eval_sols[rng.integers(len(eval_sols))]
            entries.append((n_clues, random_puzzle(sol, n_clues, rng)))
    puzzles = torch.tensor(np.stack([p for _, p in entries]), dtype=torch.long, device=device)
    boards, steps_used, _, _ = sample(model, puzzles, cfg.max_sample_steps)

    by_clue: dict[int, list[tuple[bool, int]]] = {}
    for (n_clues, puz), b, s in zip(entries, boards.cpu().numpy(), steps_used.cpu().numpy()):
        ok = solved(b, puz)
        by_clue.setdefault(n_clues, []).append((ok, int(s)))

    total_ok, total = 0, 0
    print(f"{'clues':>5} | {'n':>4} | {'solve%':>6} | mean steps (solved)")
    for n_clues in sorted(by_clue):
        rs = by_clue[n_clues]
        oks = [ok for ok, _ in rs]
        steps = [s for ok, s in rs if ok]
        total_ok += sum(oks)
        total += len(rs)
        ms = f"{np.mean(steps):5.2f}" if steps else "    -"
        print(f"{n_clues:>5} | {len(rs):>4} | {100 * np.mean(oks):6.1f} | {ms}")
    print(f"{'all':>5} | {total:>4} | {100 * total_ok / total:6.1f} |")

    rec = run_dir("records", args.model, args.training, args.seed) / "record.jsonl"
    with rec.open("a") as f:
        f.write(json.dumps({
            "type": "eval", "step": meta.get("step"), "checkpoint": meta["checkpoint"],
            "puzzle_seed": args.puzzle_seed, "n_per_clue": args.n,
            "eval_solve_rate": round(total_ok / total, 4),
            "per_clue": {str(k): round(float(np.mean([ok for ok, _ in v])), 4) for k, v in by_clue.items()},
        }) + "\n")


if __name__ == "__main__":
    main()
