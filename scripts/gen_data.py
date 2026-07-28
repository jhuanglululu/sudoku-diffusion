"""Generate the fixed eval puzzle set: uv run scripts/gen_data.py"""

from pathlib import Path

from sudoku_diffusion.data import generate_eval_set

if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "datasets" / "eval.jsonl"
    n = generate_eval_set(path)
    print(f"wrote {n} puzzles to {path}")
