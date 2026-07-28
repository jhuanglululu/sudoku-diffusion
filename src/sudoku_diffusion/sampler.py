"""Iterative masked-diffusion sampler with remasking.

Each step, for every board:
- filled non-clue cells may be *remasked*: greedy if argmax == MASK,
  stochastic with prob p(MASK);
- masked cells: the top `commit_frac` most-confident get committed to a digit
  (argmax, or temperature-sampled from the digit distribution).
Clue cells are frozen. A board terminates when it is full and no cell was
changed in a step (stable), or at max_steps. Remasking can therefore revise a
board even after it is completely filled.

For GRPO, `Rollout` records every stochastic decision so `action_logprob` can
recompute the trajectory log-probability under the current policy with grad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .data import MASK


@dataclass
class Rollout:
    puzzle: torch.Tensor                 # (16,)
    temperature: float = 1.0             # digit-sampling temperature used at rollout time
    states: list[torch.Tensor] = field(default_factory=list)   # board before each step
    remask_sites: list[torch.Tensor] = field(default_factory=list)  # bool (16,)
    remask_taken: list[torch.Tensor] = field(default_factory=list)  # bool (16,)
    commit_sites: list[torch.Tensor] = field(default_factory=list)  # bool (16,)
    commit_tokens: list[torch.Tensor] = field(default_factory=list) # long (16,)
    final_board: torch.Tensor | None = None
    steps_used: int = 0
    done: bool = False                   # stabilized (full and unchanged) within max_steps


def _digit_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Renormalized distribution over digits 1..4 (MASK excluded). (..., 4)"""
    t = max(temperature, 1e-6)
    return torch.softmax(logits[..., 1:] / t, dim=-1)


@torch.no_grad()
def sample(
    model: torch.nn.Module,
    puzzles: torch.Tensor,
    max_steps: int,
    commit_frac: float,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
    record: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[Rollout], list[list[torch.Tensor]]]:
    """Run the sampler on a batch of puzzles (B, 16).

    Returns (final_boards, steps_used, rollouts, trajectories).
    `rollouts` is empty unless record=True; `trajectories[b]` is the list of
    board states of puzzle b including the final board (for the demo).
    """
    device = puzzles.device
    boards = puzzles.clone()
    clue = puzzles != MASK
    B = boards.shape[0]
    stochastic = temperature > 0
    rollouts = [
        Rollout(puzzle=puzzles[b].clone(), temperature=temperature if stochastic else 1.0)
        for b in range(B)
    ]
    steps_used = torch.full((B,), max_steps, dtype=torch.long, device=device)
    active = torch.ones(B, dtype=torch.bool, device=device)
    trajectories: list[list[torch.Tensor]] = [[boards[b].clone()] for b in range(B)]

    for step in range(max_steps):
        if not active.any():
            break
        logits = model(boards)
        probs = torch.softmax(logits, dim=-1)
        prev = boards.clone()

        # --- remask filled non-clue cells
        filled = (boards != MASK) & ~clue & active[:, None]
        p_mask = probs[..., MASK]
        if stochastic:
            u = torch.rand(p_mask.shape, device=device, generator=generator)
            remask = filled & (u < p_mask)
        else:
            remask = filled & (logits.argmax(-1) == MASK)
        boards[remask] = MASK

        # --- commit most-confident masked cells (from pre-remask mask set)
        masked = (prev == MASK) & active[:, None]
        dprobs = _digit_probs(logits, temperature if stochastic else 1.0)
        conf = dprobs.max(-1).values.masked_fill(~masked, -1.0)
        commit = torch.zeros_like(masked)
        chosen_tok = torch.zeros_like(boards)
        for b in range(B):
            n_masked = int(masked[b].sum())
            if n_masked == 0:
                continue
            k = max(1, math.ceil(commit_frac * n_masked))
            idx = conf[b].topk(k).indices
            commit[b, idx] = True
        if stochastic:
            flat = dprobs[commit]
            tok = torch.multinomial(flat, 1, generator=generator).squeeze(-1) + 1
        else:
            tok = dprobs[commit].argmax(-1) + 1
        chosen_tok[commit] = tok
        boards[commit] = chosen_tok[commit]

        if record:
            for b in torch.nonzero(active).flatten().tolist():
                r = rollouts[b]
                r.states.append(prev[b].clone())
                r.remask_sites.append(filled[b].clone())
                r.remask_taken.append(remask[b].clone())
                r.commit_sites.append(commit[b].clone())
                r.commit_tokens.append(chosen_tok[b].clone())

        # --- termination: full and unchanged
        stable = ((boards != MASK).all(-1)) & (boards == prev).all(-1) & active
        steps_used[stable] = step + 1
        active = active & ~stable
        for b in range(B):
            if not (boards[b] == trajectories[b][-1]).all():
                trajectories[b].append(boards[b].clone())

    for b in range(B):
        rollouts[b].final_board = boards[b].clone()
        rollouts[b].steps_used = int(steps_used[b])
        rollouts[b].done = bool(~active[b])  # stabilized, even if on the last step
    return boards, steps_used, (rollouts if record else []), trajectories


def action_logprob(model: torch.nn.Module, rollouts: list[Rollout]) -> torch.Tensor:
    """Total log-probability of each rollout's recorded actions under the
    current model. Differentiable. Returns (B,)."""
    device = next(model.parameters()).device
    states, owner = [], []
    for i, r in enumerate(rollouts):
        states += r.states
        owner += [i] * len(r.states)
    if not states:
        return torch.zeros(len(rollouts), device=device)
    logits = model(torch.stack(states).to(device))
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    total = torch.zeros(len(rollouts), device=device)
    row = 0
    for i, r in enumerate(rollouts):
        for t in range(len(r.states)):
            lp, pr = logp[row], p[row]
            sites, taken = r.remask_sites[t], r.remask_taken[t]
            # remask decision: log p(MASK) if taken, log(1 - p(MASK)) otherwise
            keep = sites & ~taken
            total[i] = total[i] + lp[taken, MASK].sum()
            total[i] = total[i] + torch.log1p(-pr[keep, MASK].clamp(max=1 - 1e-6)).sum()
            # committed digits: renormalized over digits, at the rollout's temperature
            c = r.commit_sites[t]
            if c.any():
                t_ = max(r.temperature, 1e-6)
                dlp = torch.log_softmax(logits[row][c][:, 1:] / t_, dim=-1)
                total[i] = total[i] + dlp.gather(1, (r.commit_tokens[t][c] - 1)[:, None]).sum()
            row += 1
    return total
