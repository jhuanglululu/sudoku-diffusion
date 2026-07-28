import torch

from sudoku_diffusion.data import MASK
from sudoku_diffusion.sampler import action_logprob, sample


class Rigged(torch.nn.Module):
    """Returns fixed logits regardless of input."""

    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, tokens):
        return self.logits[None].expand(tokens.shape[0], -1, -1) + self.p * 0


class RiggedSeq(torch.nn.Module):
    """Returns logits_list[call_index] on successive calls (last repeats)."""

    def __init__(self, logits_list):
        super().__init__()
        self.logits_list = logits_list
        self.calls = 0
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, tokens):
        logits = self.logits_list[min(self.calls, len(self.logits_list) - 1)]
        self.calls += 1
        return logits[None].expand(tokens.shape[0], -1, -1) + self.p * 0


def test_clues_frozen_and_fills():
    # model strongly prefers digit 2 everywhere
    logits = torch.full((16, 5), -10.0)
    logits[:, 2] = 10.0
    puzzles = torch.zeros(1, 16, dtype=torch.long)
    puzzles[0, 0] = 1  # clue
    boards, steps, _, _ = sample(Rigged(logits), puzzles, max_steps=12)
    assert boards[0, 0] == 1  # clue untouched
    assert (boards[0, 1:] == 2).all()  # everything else filled with 2


def test_masked_cells_wait_on_mask_argmax():
    # cells 0..7 prefer a digit, 8..15 prefer MASK: greedy commits exactly the
    # digit-preferring cells and waits forever on the rest
    logits = torch.full((16, 5), -10.0)
    logits[:8, 4] = 10.0
    logits[8:, MASK] = 10.0
    puz = torch.zeros(1, 16, dtype=torch.long)
    boards, steps, rolls, _ = sample(Rigged(logits), puz, max_steps=5, record=True)
    assert (boards[0, :8] == 4).all()
    assert (boards[0, 8:] == MASK).all()  # never committed
    assert steps[0] == 5  # never full, runs to max_steps
    r = rolls[0]
    assert (r.mask_tokens[0][:8] == 4).all() and (r.mask_tokens[0][8:] == MASK).all()
    assert (r.mask_tokens[1][8:] == MASK).all()  # still waiting


def test_remask_happens_on_filled_board():
    # step 0: model prefers digit 3 -> fills every open cell; from step 1 it
    # prefers MASK -> must remask its own commits (but never the clue)
    fill = torch.full((16, 5), -10.0)
    fill[:, 3] = 10.0
    unfill = torch.full((16, 5), -10.0)
    unfill[:, MASK] = 10.0
    puz = torch.zeros(1, 16, dtype=torch.long)
    puz[0, 0] = 4
    boards, steps, rolls, _ = sample(RiggedSeq([fill, unfill]), puz, max_steps=3, record=True)
    assert rolls[0].remask_taken[1].any()
    assert boards[0, 0] == 4  # clue never remasked


def test_terminates_early_when_stable():
    logits = torch.full((16, 5), -10.0)
    logits[:, 1] = 10.0
    puz = torch.zeros(2, 16, dtype=torch.long)
    boards, steps, _, _ = sample(Rigged(logits), puz, max_steps=12)
    # fills everything in step 1, stable at step 2
    assert (steps == 2).all()


def test_done_when_stable_on_last_step():
    logits = torch.full((16, 5), -10.0)
    logits[:, 1] = 10.0
    puz = torch.zeros(1, 16, dtype=torch.long)
    _, steps, rolls, _ = sample(Rigged(logits), puz, max_steps=2, record=True)
    assert int(steps[0]) == 2
    assert rolls[0].done  # stabilized exactly on the last step still counts


class Regretful(torch.nn.Module):
    """Prefers digit 3 for any cell (second choice 1) but wants to remask any
    filled 3 — the deterministic commit/remask 2-cycle in miniature."""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, tokens):
        logits = torch.full((*tokens.shape, 5), -10.0)
        logits[..., 3] = 5.0
        logits[..., 1] = 2.0
        logits[tokens == 3] = torch.tensor([10.0, -10.0, -10.0, -10.0, -10.0])
        return logits + self.p * 0


def test_ban_breaks_commit_remask_cycle():
    # without the one-step ban this model loops 3 -> remask -> 3 forever;
    # with it, the second-best digit 1 commits and the board stabilizes
    puz = torch.zeros(1, 16, dtype=torch.long)
    boards, steps, rolls, _ = sample(Regretful(), puz, max_steps=12, record=True)
    assert (boards[0] == 1).all()
    assert int(steps[0]) == 4  # commit 3s, remask all, commit 1s, confirm
    r = rolls[0]
    assert (r.mask_bans[2] == 3).all()  # every cell banned from re-committing 3
    assert (r.mask_tokens[2] == 1).all()


def test_recorded_actions_consistent():
    # stochastic rollout: recorded 5-way choices must be masked-site-only and
    # exactly explain which cells got committed
    torch.manual_seed(3)
    logits = torch.randn(16, 5)
    puz = torch.zeros(1, 16, dtype=torch.long)
    puz[0, :4] = torch.tensor([1, 2, 3, 4])
    gen = torch.Generator().manual_seed(0)
    _, _, rolls, _ = sample(Rigged(logits), puz, max_steps=6, temperature=1.0, generator=gen, record=True)
    r = rolls[0]
    assert len(r.states) > 0
    for t in range(len(r.states)):
        masked = r.states[t] == MASK
        assert (r.mask_sites[t] == masked).all()
        assert (r.mask_tokens[t][~masked] == 0).all()
        committed = masked & (r.mask_tokens[t] != MASK)
        assert (r.mask_tokens[t][r.mask_bans[t] > 0] != r.mask_bans[t][r.mask_bans[t] > 0]).all()
        if t + 1 < len(r.states):
            nxt = r.states[t + 1]
            assert (nxt[committed] == r.mask_tokens[t][committed]).all()
            # a cell remasked at t carries a ban on the removed digit at t+1
            remasked = r.remask_taken[t]
            assert (r.mask_bans[t + 1][remasked] == r.states[t][remasked]).all()


def _manual_logprob(logits, roll, temperature):
    logp = torch.log_softmax(logits, -1)
    p = logp.exp()
    total = 0.0
    for t in range(len(roll.states)):
        taken, keep = roll.remask_taken[t], roll.remask_sites[t] & ~roll.remask_taken[t]
        total += logp[taken, MASK].sum() + torch.log1p(-p[keep, MASK]).sum()
        # 5-way choice renormalized over non-banned tokens, like the sampler
        banned_logits = logits.clone()
        bans = roll.mask_bans[t]
        banned_logits[bans > 0, bans[bans > 0]] = float("-inf")
        tlp = torch.log_softmax(banned_logits / temperature, -1)
        m = roll.mask_sites[t]
        if m.any():
            total += tlp[m].gather(1, roll.mask_tokens[t][m][:, None]).sum()
    return total


def test_action_logprob_matches_manual():
    torch.manual_seed(0)
    logits = torch.randn(16, 5)
    model = Rigged(logits)
    puz = torch.zeros(1, 16, dtype=torch.long)
    gen = torch.Generator().manual_seed(0)
    _, _, rolls, _ = sample(model, puz, max_steps=4, temperature=1.0, generator=gen, record=True)
    lp = action_logprob(model, rolls)
    assert torch.isfinite(lp).all()
    assert torch.allclose(lp[0], _manual_logprob(logits, rolls[0], 1.0), atol=1e-5)


def test_action_logprob_multiple_rollouts():
    torch.manual_seed(2)
    logits = torch.randn(16, 5)
    model = Rigged(logits)
    puz = torch.zeros(2, 16, dtype=torch.long)
    puz[1, :4] = torch.tensor([1, 2, 3, 4])  # different clue counts -> different rollouts
    gen = torch.Generator().manual_seed(1)
    _, _, rolls, _ = sample(model, puz, max_steps=5, temperature=1.0, generator=gen, record=True)
    lp = action_logprob(model, rolls)
    for i in range(2):
        assert torch.allclose(lp[i], _manual_logprob(logits, rolls[i], 1.0), atol=1e-5)


def test_action_logprob_applies_temperature():
    torch.manual_seed(0)
    logits = torch.randn(16, 5)
    model = Rigged(logits)
    puz = torch.zeros(1, 16, dtype=torch.long)
    gen = torch.Generator().manual_seed(0)
    _, _, rolls, _ = sample(model, puz, max_steps=4, temperature=0.7, generator=gen, record=True)
    assert rolls[0].temperature == 0.7
    lp = action_logprob(model, rolls)
    assert torch.allclose(lp[0], _manual_logprob(logits, rolls[0], 0.7), atol=1e-5)
