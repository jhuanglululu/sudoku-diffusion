import math

import torch

from sudoku_diffusion.data import MASK
from sudoku_diffusion.sampler import Rollout, _digit_probs, action_logprob, sample


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
    # Non-clue filled cells only exist mid-trajectory, so start from a 1-clue
    # puzzle: step 0 commits digit 3 to every open cell, and a MASK-preferring
    # model must remask those own commits on step 1 (but never the clue).
    logits = torch.full((16, 5), -10.0)
    logits[:, MASK] = 10.0
    logits[:, 3] += 5.0  # digit fallback so commits pick 3
    puz = torch.zeros(1, 16, dtype=torch.long)
    puz[0, 0] = 4
    boards, steps, rolls, _ = sample(Rigged(logits), puz, max_steps=3, commit_frac=1.0, record=True)
    assert rolls[0].remask_taken[1].any()
    assert boards[0, 0] == 4  # clue never remasked


def test_terminates_early_when_stable():
    logits = torch.full((16, 5), -10.0)
    logits[:, 1] = 10.0
    puz = torch.zeros(2, 16, dtype=torch.long)
    boards, steps, _, _ = sample(Rigged(logits), puz, max_steps=12, commit_frac=1.0)
    # fills everything in step 1, stable at step 2
    assert (steps == 2).all()


def test_done_when_stable_on_last_step():
    logits = torch.full((16, 5), -10.0)
    logits[:, 1] = 10.0
    puz = torch.zeros(1, 16, dtype=torch.long)
    _, steps, rolls, _ = sample(Rigged(logits), puz, max_steps=2, commit_frac=1.0, record=True)
    assert int(steps[0]) == 2
    assert rolls[0].done  # stabilized exactly on the last step still counts


def _manual_logprob(logits, roll, temperature):
    logp = torch.log_softmax(logits, -1)
    p = logp.exp()
    dlp = torch.log_softmax(logits[:, 1:] / temperature, -1)
    total = 0.0
    for t in range(len(roll.states)):
        taken, keep = roll.remask_taken[t], roll.remask_sites[t] & ~roll.remask_taken[t]
        total += logp[taken, MASK].sum() + torch.log1p(-p[keep, MASK]).sum()
        c = roll.commit_sites[t]
        if c.any():
            total += dlp[c].gather(1, (roll.commit_tokens[t][c] - 1)[:, None]).sum()
    return total


def test_action_logprob_matches_manual():
    torch.manual_seed(0)
    logits = torch.randn(16, 5)
    model = Rigged(logits)
    puz = torch.zeros(1, 16, dtype=torch.long)
    gen = torch.Generator().manual_seed(0)
    _, _, rolls, _ = sample(model, puz, max_steps=4, commit_frac=0.5, temperature=1.0, generator=gen, record=True)
    lp = action_logprob(model, rolls)
    assert torch.isfinite(lp).all()
    assert torch.allclose(lp[0], _manual_logprob(logits, rolls[0], 1.0), atol=1e-5)


def test_commit_selection_is_topk_of_masked():
    # tie-free random logits; per recorded step, the committed set must be
    # masked-only, sized max(1, ceil(frac * n_masked)), and dominate every
    # non-committed masked cell in confidence
    torch.manual_seed(1)
    logits = torch.randn(16, 5)
    conf = _digit_probs(logits, 1.0).max(-1).values
    frac = 0.35
    puz = torch.zeros(3, 16, dtype=torch.long)
    puz[1, :4] = torch.tensor([1, 2, 3, 4])
    puz[2, 0] = 2
    _, _, rolls, _ = sample(Rigged(logits), puz, max_steps=6, commit_frac=frac, record=True)
    checked = 0
    for r in rolls:
        for t in range(len(r.states)):
            masked = r.states[t] == MASK
            commit = r.commit_sites[t]
            n_masked = int(masked.sum())
            if n_masked == 0:
                assert not commit.any()
                continue
            assert (commit & ~masked).sum() == 0
            assert int(commit.sum()) == max(1, math.ceil(frac * n_masked))
            open_conf = conf[masked & ~commit]
            if commit.any() and open_conf.numel():
                assert conf[commit].min() >= open_conf.max()
            checked += 1
    assert checked > 0


def test_action_logprob_multiple_rollouts():
    torch.manual_seed(2)
    logits = torch.randn(16, 5)
    model = Rigged(logits)
    puz = torch.zeros(2, 16, dtype=torch.long)
    puz[1, :4] = torch.tensor([1, 2, 3, 4])  # different clue counts -> different rollouts
    gen = torch.Generator().manual_seed(1)
    _, _, rolls, _ = sample(model, puz, max_steps=5, commit_frac=0.5, temperature=1.0, generator=gen, record=True)
    lp = action_logprob(model, rolls)
    for i in range(2):
        assert torch.allclose(lp[i], _manual_logprob(logits, rolls[i], 1.0), atol=1e-5)


def test_action_logprob_applies_temperature():
    torch.manual_seed(0)
    logits = torch.randn(16, 5)
    model = Rigged(logits)
    puz = torch.zeros(1, 16, dtype=torch.long)
    gen = torch.Generator().manual_seed(0)
    _, _, rolls, _ = sample(model, puz, max_steps=4, commit_frac=0.5, temperature=0.7, generator=gen, record=True)
    assert rolls[0].temperature == 0.7
    lp = action_logprob(model, rolls)
    assert torch.allclose(lp[0], _manual_logprob(logits, rolls[0], 0.7), atol=1e-5)
