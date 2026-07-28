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
    track_trajectories: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[Rollout], list[list[torch.Tensor]]]:
    """Run the sampler on a batch of puzzles (B, 16).

    Returns (final_boards, steps_used, rollouts, trajectories).
    `rollouts` is empty unless record=True; `trajectories` is empty unless
    track_trajectories=True, else `trajectories[b]` is the list of board
    states of puzzle b including the final board (for the demo).
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
    trajectories: list[list[torch.Tensor]] = (
        [[boards[b].clone()] for b in range(B)] if track_trajectories else []
    )
    # per-step batched records; sliced into per-board Rollout lists at the end
    rec: list[tuple[torch.Tensor, ...]] = []

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
        # per-board k = max(1, ceil(commit_frac * n_masked)), 0 if nothing masked;
        # commit the cells whose confidence rank is below k. Non-masked cells sit
        # at conf -1.0 < any probability, so ranks < n_masked are all masked cells.
        # k in float64 so the ceil matches math.ceil at boundary values.
        n_masked = masked.sum(-1)
        k = (n_masked.double() * commit_frac).ceil().long().clamp(min=1)
        k = torch.where(n_masked > 0, k, torch.zeros_like(k))
        ranks = conf.argsort(-1, descending=True).argsort(-1)
        commit = ranks < k[:, None]
        chosen_tok = torch.zeros_like(boards)
        if stochastic:
            flat = dprobs[commit]
            tok = torch.multinomial(flat, 1, generator=generator).squeeze(-1) + 1
        else:
            tok = dprobs[commit].argmax(-1) + 1
        chosen_tok[commit] = tok
        boards[commit] = chosen_tok[commit]

        if record:
            rec.append((active.clone(), prev, filled, remask, commit, chosen_tok))

        # --- termination: full and unchanged
        stable = ((boards != MASK).all(-1)) & (boards == prev).all(-1) & active
        steps_used[stable] = step + 1
        active = active & ~stable
        if track_trajectories:
            for b in range(B):
                if not (boards[b] == trajectories[b][-1]).all():
                    trajectories[b].append(boards[b].clone())

    if record:
        act = torch.stack([a for a, *_ in rec]).cpu().tolist() if rec else []
        for b in range(B):
            r = rollouts[b]
            for t, (_, prev, filled, remask, commit, chosen_tok) in enumerate(rec):
                if not act[t][b]:
                    break  # once inactive, a board never records again
                r.states.append(prev[b])
                r.remask_sites.append(filled[b])
                r.remask_taken.append(remask[b])
                r.commit_sites.append(commit[b])
                r.commit_tokens.append(chosen_tok[b])
    for b in range(B):
        rollouts[b].final_board = boards[b].clone()
        rollouts[b].steps_used = int(steps_used[b])
        rollouts[b].done = bool(~active[b])  # stabilized, even if on the last step
    return boards, steps_used, (rollouts if record else []), trajectories


def action_logprob(model: torch.nn.Module, rollouts: list[Rollout]) -> torch.Tensor:
    """Total log-probability of each rollout's recorded actions under the
    current model. Differentiable. Returns (B,).

    Fully batched: all recorded steps of all rollouts go through one forward
    and a handful of tensor ops, so the autograd graph stays small."""
    device = next(model.parameters()).device
    states, sites, taken, csites, ctoks, owner, temps = [], [], [], [], [], [], []
    for i, r in enumerate(rollouts):
        states += r.states
        sites += r.remask_sites
        taken += r.remask_taken
        csites += r.commit_sites
        ctoks += r.commit_tokens
        owner += [i] * len(r.states)
        temps += [max(r.temperature, 1e-6)] * len(r.states)
    if not states:
        return torch.zeros(len(rollouts), device=device)
    sites_t = torch.stack(sites).to(device)
    taken_t = torch.stack(taken).to(device)
    commit = torch.stack(csites).to(device)                       # (N, 16) bool
    tokens = torch.stack(ctoks).to(device)                        # (N, 16) long
    temp = torch.tensor(temps, device=device)[:, None, None]      # (N, 1, 1)

    logits = model(torch.stack(states).to(device))                # (N, 16, 5)
    logp = torch.log_softmax(logits, dim=-1)
    p_mask = logp[..., MASK].exp()
    # remask decision: log p(MASK) if taken, log(1 - p(MASK)) otherwise
    keep = sites_t & ~taken_t
    per_state = (logp[..., MASK] * taken_t).sum(-1)
    per_state = per_state + (torch.log1p(-p_mask.clamp(max=1 - 1e-6)) * keep).sum(-1)
    # committed digits: renormalized over digits, at the rollout's temperature
    dlp = torch.log_softmax(logits[..., 1:] / temp, dim=-1)       # (N, 16, 4)
    digit_lp = dlp.gather(-1, (tokens - 1).clamp(min=0)[..., None]).squeeze(-1)
    per_state = per_state + (digit_lp * commit).sum(-1)

    owner_t = torch.tensor(owner, device=device)
    return torch.zeros(len(rollouts), device=device).index_add(0, owner_t, per_state)
