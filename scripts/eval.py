"""Greedy solve-rate eval on freshly generated eval-orbit puzzles.

uv run scripts/eval.py --model tiny --training sft --seed 0 --puzzle-seed 0 --n 100
"""

import argparse
import json

import numpy as np
import torch

from sudoku_diffusion.data import MASK, SPLIT_SEED, orbit_split, random_puzzle, scrambled_board, solved
from sudoku_diffusion.model import SudokuDenoiser, get_device
from sudoku_diffusion.runs import load_checkpoint, run_dir
from sudoku_diffusion.sampler import sample
from sudoku_diffusion.variations import MODELS, TRAININGS

CLUE_COUNTS = (0, 1, 2, 3, 4, 5, 6, 7, 8)
SCRAM = -1  # row label for scrambled full-board starts


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
    meta = load_checkpoint(model, args.model, args.training, args.seed, which="current")
    print(f"loaded {meta['checkpoint']} checkpoint (step {meta.get('step')})")
    model.eval()

    rng = np.random.default_rng(args.puzzle_seed)
    _, eval_sols = orbit_split(np.random.default_rng(SPLIT_SEED))
    entries: list[tuple[int, np.ndarray, np.ndarray]] = []  # (key, puzzle, init)
    for n_clues in CLUE_COUNTS:
        if n_clues == 0:
            empty = np.zeros(16, dtype=eval_sols.dtype)
            entries.append((0, empty, empty))
            continue
        for _ in range(args.n):
            sol = eval_sols[rng.integers(len(eval_sols))]
            puz = random_puzzle(sol, n_clues, rng)
            entries.append((n_clues, puz, puz))
    for _ in range(args.n):  # clue-free scrambled starts: repair from garbage
        entries.append((SCRAM, np.zeros(16, dtype=eval_sols.dtype), scrambled_board(rng)))
    puzzles = torch.tensor(np.stack([p for _, p, _ in entries]), dtype=torch.long, device=device)
    inits = torch.tensor(np.stack([i for _, _, i in entries]), dtype=torch.long, device=device)
    gen = torch.Generator(device=device).manual_seed(args.puzzle_seed)
    boards, steps_used, rollouts, _ = sample(
        model, puzzles, cfg.max_sample_steps, record=True,
        warmup_steps=cfg.sample_warmup_steps, generator=gen, init_boards=inits,
    )

    # failure modes: stable-invalid = model declared done (full and unchanged)
    # on an invalid board; thrash = still flip-flopping at max_steps with a
    # full board; stall = never filled the board (kept waiting)
    def classify(ok: bool, done: bool, full: bool) -> str:
        if ok:
            return "solved"
        if done:
            return "stable_invalid"
        return "thrash" if full else "stall"

    by_clue: dict[int, list[tuple[str, int]]] = {}
    for (key, puz, _), b, s, r in zip(entries, boards.cpu().numpy(), steps_used.cpu().numpy(), rollouts):
        cat = classify(solved(b, puz), r.done, bool((b != MASK).all()))
        by_clue.setdefault(key, []).append((cat, int(s)))

    total_ok, total = 0, 0
    fail_totals = {"stable_invalid": 0, "thrash": 0, "stall": 0}
    print(f"{'clues':>5} | {'n':>4} | {'solve%':>6} | {'steps':>5} | {'st-inv':>6} | {'thrash':>6} | {'stall':>5}")
    for key in sorted(by_clue, key=lambda k: (k == SCRAM, k)):  # scram row last
        rs = by_clue[key]
        oks = [cat == "solved" for cat, _ in rs]
        steps = [s for cat, s in rs if cat == "solved"]
        counts = {k: sum(cat == k for cat, _ in rs) for k in fail_totals}
        for k, v in counts.items():
            fail_totals[k] += v
        total_ok += sum(oks)
        total += len(rs)
        ms = f"{np.mean(steps):5.2f}" if steps else "    -"
        label = "scram" if key == SCRAM else str(key)
        print(
            f"{label:>5} | {len(rs):>4} | {100 * np.mean(oks):6.1f} | {ms} | "
            f"{counts['stable_invalid']:>6} | {counts['thrash']:>6} | {counts['stall']:>5}"
        )
    print(
        f"{'all':>5} | {total:>4} | {100 * total_ok / total:6.1f} | {'':>5} | "
        f"{fail_totals['stable_invalid']:>6} | {fail_totals['thrash']:>6} | {fail_totals['stall']:>5}"
    )

    rec = run_dir("records", args.model, args.training, args.seed) / "record.jsonl"
    with rec.open("a") as f:
        f.write(json.dumps({
            "type": "eval", "step": meta.get("step"), "checkpoint": meta["checkpoint"],
            "puzzle_seed": args.puzzle_seed, "n_per_clue": args.n,
            "eval_solve_rate": round(total_ok / total, 4),
            "per_clue": {
                ("scram" if k == SCRAM else str(k)): round(float(np.mean([cat == "solved" for cat, _ in v])), 4)
                for k, v in by_clue.items()
            },
            "failures": fail_totals,
        }) + "\n")


if __name__ == "__main__":
    main()
