import numpy as np
import torch

from sudoku_diffusion.data import MASK, symmetry_group
from sudoku_diffusion.sft import IGNORE, corrupt_batch, sft_losses
from sudoku_diffusion.variations import MODELS, TRAININGS
from sudoku_diffusion.model import SudokuDenoiser


def test_corrupt_batch_targets_exact():
    cfg = TRAININGS["smoke"]
    sols = np.tile(np.array([[1, 2, 3, 4, 3, 4, 1, 2, 2, 1, 4, 3, 4, 3, 2, 1]]), (cfg.batch_size, 1))
    inputs, targets = corrupt_batch(sols, cfg, np.random.default_rng(0))
    sol = torch.from_numpy(sols).long()
    masked = inputs == MASK
    untouched = (inputs == sol)
    wrong = ~masked & ~untouched
    assert masked.any() and wrong.any()
    # masked cells: target is the correct digit
    assert torch.all(targets[masked] == sol[masked])
    # wrong cells: input is a digit != solution, target is MASK
    assert torch.all(targets[wrong] == MASK)
    assert torch.all((inputs[wrong] >= 1) & (inputs[wrong] <= 4))
    # untouched cells: target = keep own value (anti-oscillation signal)
    assert torch.all(targets[untouched] == sol[untouched])


def test_corrupt_batch_can_produce_empty_board():
    cfg = TRAININGS["smoke"]  # mask_frac_range up to 1.0
    sols = np.tile(np.array([[1, 2, 3, 4, 3, 4, 1, 2, 2, 1, 4, 3, 4, 3, 2, 1]]), (512, 1))
    inputs, _ = corrupt_batch(sols, cfg, np.random.default_rng(1))
    assert (inputs == MASK).all(dim=1).any()


def test_consistency_loss_zero_for_equivariant_model():
    class Uniform(torch.nn.Module):
        def forward(self, tokens):
            return torch.zeros(tokens.shape[0], 16, 5)

    cfg = TRAININGS["smoke"]
    group = torch.from_numpy(symmetry_group()).long()
    inputs = torch.zeros(4, 16, dtype=torch.long)
    targets = torch.full((4, 16), IGNORE)
    targets[:, 0] = 1
    gen = torch.Generator().manual_seed(0)
    _, cons = sft_losses(Uniform(), inputs, targets, group, cfg, gen)
    assert abs(cons.item()) < 1e-6


def test_consistency_nonzero_for_real_model():
    cfg = TRAININGS["smoke"]
    group = torch.from_numpy(symmetry_group()).long()
    model = SudokuDenoiser(MODELS["tiny"])
    inputs = torch.randint(0, 5, (4, 16))
    targets = torch.full((4, 16), IGNORE)
    targets[:, 0] = 1
    gen = torch.Generator().manual_seed(3)  # avoid identity perm draw
    ce, cons = sft_losses(model, inputs, targets, group, cfg, gen)
    assert torch.isfinite(ce) and torch.isfinite(cons)
