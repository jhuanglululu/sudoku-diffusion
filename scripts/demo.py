"""Show the diffusion/remasking trajectory for one puzzle.

uv run scripts/demo.py --model tiny --training smoke --seed 0 [--clues 5]
--clues 0 starts from a completely empty board.
"""

import argparse

import numpy as np
import torch

from sudoku_diffusion.data import MASK, SPLIT_SEED, make_puzzle, orbit_split, random_puzzle, solved
from sudoku_diffusion.model import SudokuDenoiser, get_device
from sudoku_diffusion.runs import load_checkpoint
from sudoku_diffusion.sampler import sample
from sudoku_diffusion.variations import MODELS, TRAININGS


def show(board: np.ndarray, prev: np.ndarray | None) -> str:
    lines = []
    for r in range(4):
        row = []
        for c in range(4):
            i = r * 4 + c
            ch = "." if board[i] == MASK else str(board[i])
            if prev is not None and board[i] != prev[i]:
                ch = f"\033[93m{ch}\033[0m"  # changed cell in yellow
            row.append(ch)
            if c == 1:
                row.append("|")
        lines.append(" ".join(row))
        if r == 1:
            lines.append("----+----")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--training", required=True, choices=sorted(TRAININGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clues", type=int, default=5)
    ap.add_argument("--puzzle-seed", type=int, default=0)
    args = ap.parse_args()
    cfg = TRAININGS[args.training]

    rng = np.random.default_rng(args.puzzle_seed)
    if args.clues == 0:
        puzzle = np.zeros(16, dtype=np.int64)
    else:
        _, eval_sols = orbit_split(np.random.default_rng(SPLIT_SEED))
        if args.clues < 4:
            # no unique puzzle exists below 4 clues; blank at random instead
            # (solved() verifies the board, so any valid completion counts)
            puzzle = random_puzzle(eval_sols[rng.integers(len(eval_sols))], args.clues, rng)
        else:
            puzzle = None
            while puzzle is None:
                puzzle = make_puzzle(eval_sols[rng.integers(len(eval_sols))], args.clues, rng)

    device = get_device()
    model = SudokuDenoiser(MODELS[args.model]).to(device)
    load_checkpoint(model, args.model, args.training, args.seed)
    model.eval()

    boards, steps_used, _, trajs = sample(
        model, torch.from_numpy(puzzle)[None].long().to(device), cfg.max_sample_steps,
        track_trajectories=True,
    )
    traj = [t.cpu().numpy() for t in trajs[0]]
    print(f"puzzle ({args.clues} clues):")
    for i, b in enumerate(traj):
        label = "start" if i == 0 else f"step {i}"
        print(f"\n--- {label} ---")
        print(show(b, traj[i - 1] if i else None))
    final = boards[0].cpu().numpy()
    print(f"\nsolved: {solved(final, puzzle)}  steps used: {int(steps_used[0])}")


if __name__ == "__main__":
    main()
