import torch

from sudoku_diffusion.data import BOXES, COLS, ROWS, symmetry_group
from sudoku_diffusion.model import SudokuDenoiser, _mask_from_units
from sudoku_diffusion.variations import ModelConfig

TINY_UNIT = ModelConfig(name="t", d_model=32, n_layers=2, n_heads=2, d_ff=64, arch="unit")
TINY_TF = ModelConfig(name="t", d_model=32, n_layers=2, n_heads=2, d_ff=64)


def test_unit_masks():
    row, col, box = (_mask_from_units(u) for u in (ROWS, COLS, BOXES))
    for m in (row, col, box):
        assert (m.sum(-1) == 3).all()  # 3 peers per unit type
        assert (m == m.T).all() and not m.diagonal().any()
    peers = row | col | box
    assert (peers.sum(-1) == 7).all()  # 3 row + 3 col + 1 box-diagonal
    assert set(peers[0].nonzero().flatten().tolist()) == {1, 2, 3, 4, 5, 8, 12}


def test_forward_shapes_both_archs():
    tokens = torch.randint(0, 5, (3, 16))
    for cfg in (TINY_TF, TINY_UNIT):
        logits = SudokuDenoiser(cfg)(tokens)
        assert logits.shape == (3, 16, 5)
    assert not hasattr(SudokuDenoiser(TINY_UNIT), "pos")


def test_unit_arch_is_exactly_equivariant():
    # model(x[g]) == model(x)[g] for every geometric symmetry, at random
    # (untrained) weights — the property holds by construction, not training
    torch.manual_seed(0)
    model = SudokuDenoiser(TINY_UNIT).eval()
    tokens = torch.randint(0, 5, (2, 16))
    with torch.no_grad():
        base = model(tokens)
        for g in torch.from_numpy(symmetry_group()).long()[::17]:  # sample of the 128
            assert torch.allclose(model(tokens[:, g]), base[:, g], atol=1e-5)


def test_unit_arch_symmetric_board_degenerate():
    # equivariance corollary: on the empty board every cell gets identical
    # logits — this is why greedy sampling needs the stochastic warmup
    torch.manual_seed(0)
    model = SudokuDenoiser(TINY_UNIT).eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 16, dtype=torch.long))
    assert torch.allclose(logits, logits[:, :1].expand(-1, 16, -1), atol=1e-5)
