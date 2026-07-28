import torch

from sudoku_diffusion.data import MASK
from sudoku_diffusion.sampler import Rollout, action_logprob, sample


class Rigged(torch.nn.Module):
    """Returns fixed logits regardless of input."""

    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, tokens):
        return self.logits[None].expand(tokens.shape[0], -1, -1) + self.p * 0


def test_clues_frozen_and_fills():
    # model strongly prefers digit 2 everywhere
    logits = torch.full((16, 5), -10.0)
    logits[:, 2] = 10.0
    puzzles = torch.zeros(1, 16, dtype=torch.long)
    puzzles[0, 0] = 1  # clue
    boards, steps, _, _ = sample(Rigged(logits), puzzles, max_steps=12, commit_frac=0.35)
    assert boards[0, 0] == 1  # clue untouched
    assert (boards[0, 1:] == 2).all()  # everything else filled with 2


def test_remask_happens_on_filled_board():
    # model wants MASK everywhere: a pre-filled wrong cell must get remasked
    logits = torch.full((16, 5), -10.0)
    logits[:, MASK] = 10.0
    logits[:, 3] += 5.0  # digit fallback so commits pick 3
    puzzles = torch.full((1, 16), 4, dtype=torch.long)
    puzzles[0, 0] = MASK  # one open cell, 15 "clues"? no — make them non-clue:
    # non-clue filled cells only exist mid-trajectory; emulate via a puzzle with
    # 1 clue, then check the model remasks its own commits on the next step.
    puz = torch.zeros(1, 16, dtype=torch.long)
    puz[0, 0] = 4
    boards, steps, rolls, _ = sample(Rigged(logits), puz, max_steps=3, commit_frac=1.0, record=True)
    # step 0 commits digit 3 everywhere; step 1 must remask those cells
    assert rolls[0].remask_taken[1].any()
    assert boards[0, 0] == 4  # clue never remasked


def test_terminates_early_when_stable():
    logits = torch.full((16, 5), -10.0)
    logits[:, 1] = 10.0
    puz = torch.zeros(2, 16, dtype=torch.long)
    boards, steps, _, _ = sample(Rigged(logits), puz, max_steps=12, commit_frac=1.0)
    # fills everything in step 1, stable at step 2
    assert (steps == 2).all()


def test_action_logprob_matches_manual():
    torch.manual_seed(0)
    logits = torch.randn(16, 5)
    model = Rigged(logits)
    puz = torch.zeros(1, 16, dtype=torch.long)
    gen = torch.Generator().manual_seed(0)
    _, _, rolls, _ = sample(model, puz, max_steps=4, commit_frac=0.5, temperature=1.0, generator=gen, record=True)
    lp = action_logprob(model, rolls)
    assert torch.isfinite(lp).all()

    # manual recomputation for the recorded actions
    logp = torch.log_softmax(logits, -1)
    p = logp.exp()
    dlp = torch.log_softmax(logits[:, 1:], -1)
    total = 0.0
    r = rolls[0]
    for t in range(len(r.states)):
        taken, keep = r.remask_taken[t], r.remask_sites[t] & ~r.remask_taken[t]
        total += logp[taken, MASK].sum() + torch.log1p(-p[keep, MASK]).sum()
        c = r.commit_sites[t]
        if c.any():
            total += dlp[c].gather(1, (r.commit_tokens[t][c] - 1)[:, None]).sum()
    assert torch.allclose(lp[0], total, atol=1e-5)
