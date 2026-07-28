"""Train a variation: uv run scripts/train.py --model tiny --training smoke"""

import argparse

from sudoku_diffusion.variations import MODELS, TRAININGS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--training", required=True, choices=sorted(TRAININGS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if TRAININGS[args.training].kind == "sft":
        from sudoku_diffusion.sft import train_sft

        train_sft(args.model, args.training, args.seed)
    else:
        from sudoku_diffusion.grpo import train_grpo

        train_grpo(args.model, args.training, args.seed)


if __name__ == "__main__":
    main()
