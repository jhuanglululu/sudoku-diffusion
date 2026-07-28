"""Greedy solve-rate eval on the fixed eval set.

uv run scripts/eval.py --model tiny --training smoke --seed 0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sudoku_diffusion.data import load_eval_set, solved
from sudoku_diffusion.model import SudokuDenoiser, get_device
from sudoku_diffusion.runs import load_checkpoint, run_dir
from sudoku_diffusion.sampler import sample
from sudoku_diffusion.variations import MODELS, TRAININGS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--training", required=True, choices=sorted(TRAININGS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = TRAININGS[args.training]

    device = get_device()
    model = SudokuDenoiser(MODELS[args.model]).to(device)
    load_checkpoint(model, args.model, args.training, args.seed)
    model.eval()

    entries = load_eval_set(Path(__file__).resolve().parents[1] / "datasets" / "eval.jsonl")
    puzzles = torch.tensor([e["puzzle"] for e in entries], dtype=torch.long, device=device)
    boards, steps_used, _, _ = sample(model, puzzles, cfg.max_sample_steps, cfg.commit_frac)

    by_clue: dict[int, list[tuple[bool, int]]] = {}
    for e, b, s in zip(entries, boards.cpu().numpy(), steps_used.cpu().numpy()):
        ok = solved(b, np.array(e["puzzle"]))
        by_clue.setdefault(e["n_clues"], []).append((ok, int(s)))

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
            "type": "eval", "step": -1, "eval_solve_rate": round(total_ok / total, 4),
            "per_clue": {str(k): round(float(np.mean([ok for ok, _ in v])), 4) for k, v in by_clue.items()},
        }) + "\n")


if __name__ == "__main__":
    main()
