import numpy as np
import torch

from sudoku_diffusion.data import all_solutions
from sudoku_diffusion.grpo import group_advantages, reward, sample_puzzles
from sudoku_diffusion.variations import TRAININGS


def test_group_advantages_normalized():
    r = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]])
    a = group_advantages(r)
    assert abs(a[0].mean().item()) < 1e-5
    assert a[1].abs().max().item() < 1e-2  # constant group -> ~zero advantage
    assert a[0, 3] > a[0, 0]


def test_reward_prefers_fewer_steps():
    cfg = TRAININGS["grpo-smoke"]
    sol = all_solutions()[0]
    empty = np.zeros(16, dtype=sol.dtype)
    fast, ok1 = reward(sol, empty, steps_used=3, cfg=cfg)
    slow, ok2 = reward(sol, empty, steps_used=cfg.max_sample_steps, cfg=cfg)
    assert ok1 and ok2
    assert fast > slow
    unsolved, ok3 = reward(np.ones(16, dtype=sol.dtype), empty, steps_used=3, cfg=cfg)
    assert not ok3 and unsolved < slow


def test_sample_puzzles_includes_empty():
    cfg = TRAININGS["grpo-smoke"].model_copy(update={"puzzles_per_batch": 32})
    train_sols = all_solutions()[:100]
    puzzles = sample_puzzles(train_sols, cfg, np.random.default_rng(0))
    assert puzzles.shape == (32, 16)
    assert any((p == 0).all() for p in puzzles)  # empty boards present


def test_sample_puzzles_low_clue_counts():
    # 1..3 clues never admit a unique 4x4 puzzle; generation must still work
    # because the reward verifies boards instead of matching one solution
    cfg = TRAININGS["grpo-smoke"].model_copy(
        update={"puzzles_per_batch": 64, "clue_counts": (1, 2, 3)}
    )
    train_sols = all_solutions()[:100]
    puzzles = sample_puzzles(train_sols, cfg, np.random.default_rng(0))
    n_clues = (puzzles != 0).sum(axis=1)
    assert set(n_clues.tolist()) == {1, 2, 3}
    assert ((puzzles >= 0) & (puzzles <= 4)).all()  # clues are real digits
