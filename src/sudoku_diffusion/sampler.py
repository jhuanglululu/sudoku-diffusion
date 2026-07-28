"""Iterative masked-diffusion sampler with remasking.

Each step, for every board:
- filled non-clue cells may be *remasked*: greedy if argmax == MASK,
  stochastic with prob p(MASK);
- masked cells choose among all 5 tokens (greedy argmax, or temperature-
  sampled): a digit commits the cell, MASK means "not decided yet". The
  model paces its own commits through the probability it leaves on MASK —
  there is no external commit schedule.
A cell that was just remasked cannot re-commit the digit that was removed on
the very next step (one-step ban) — this breaks the deterministic
commit/remask 2-cycle, forcing the second-best digit or a wait instead. The
ban is part of the action distribution, so rollouts record it and
action_logprob renormalizes identically.
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
    temperature: float = 1.0             # sampling temperature used at rollout time
    states: list[torch.Tensor] = field(default_factory=list)   # board before each step
    remask_sites: list[torch.Tensor] = field(default_factory=list)  # bool (16,) filled non-clue cells
    remask_taken: list[torch.Tensor] = field(default_factory=list)  # bool (16,)
    mask_sites: list[torch.Tensor] = field(default_factory=list)    # bool (16,) cells masked before the step
    mask_tokens: list[torch.Tensor] = field(default_factory=list)   # long (16,) 5-way choice there; MASK = wait
    mask_bans: list[torch.Tensor] = field(default_factory=list)     # long (16,) digit banned this step; 0 = none
    final_board: torch.Tensor | None = None
    steps_used: int = 0
    done: bool = False                   # stabilized (full and unchanged) within max_steps


def _ban_logits(logits: torch.Tensor, banned: torch.Tensor) -> torch.Tensor:
    """Exclude each cell's banned digit (0 = none; MASK is never banned)."""
    ban_mask = torch.zeros_like(logits, dtype=torch.bool)
    ban_mask.scatter_(-1, banned[..., None], True)
    ban_mask[..., MASK] = False  # the 0 sentinel lands on the MASK column
    return logits.masked_fill(ban_mask, float("-inf"))


@torch.no_grad()
def sample(
    model: torch.nn.Module,
    puzzles: torch.Tensor,
    max_steps: int,
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
    # digit banned per cell for this step's 5-way choice (0 = none): a cell
    # remasked at step t may not re-commit the removed digit at step t+1
    banned = torch.zeros_like(boards)

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

        # --- masked cells (from the pre-remask mask set): 5-way choice,
        # a digit commits the cell, MASK waits; the banned digit is excluded
        masked = (prev == MASK) & active[:, None]
        choice_logits = _ban_logits(logits, banned)
        if stochastic:
            tprobs = torch.softmax(choice_logits / max(temperature, 1e-6), dim=-1)
            tok = torch.multinomial(
                tprobs.reshape(-1, tprobs.shape[-1]), 1, generator=generator
            ).reshape(B, -1)
        else:
            tok = choice_logits.argmax(-1)
        chosen_tok = torch.where(masked, tok, torch.zeros_like(boards))
        commit = masked & (chosen_tok != MASK)
        boards[commit] = chosen_tok[commit]

        if record:
            rec.append((active.clone(), prev, filled, remask, masked, chosen_tok, banned))
        banned = torch.where(remask, prev, torch.zeros_like(prev))  # one step only

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
            for t, (_, prev, filled, remask, masked, chosen_tok, bans) in enumerate(rec):
                if not act[t][b]:
                    break  # once inactive, a board never records again
                r.states.append(prev[b])
                r.remask_sites.append(filled[b])
                r.remask_taken.append(remask[b])
                r.mask_sites.append(masked[b])
                r.mask_tokens.append(chosen_tok[b])
                r.mask_bans.append(bans[b])
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
    states, sites, taken, msites, mtoks, mbans, owner, temps = [], [], [], [], [], [], [], []
    for i, r in enumerate(rollouts):
        states += r.states
        sites += r.remask_sites
        taken += r.remask_taken
        msites += r.mask_sites
        mtoks += r.mask_tokens
        mbans += r.mask_bans
        owner += [i] * len(r.states)
        temps += [max(r.temperature, 1e-6)] * len(r.states)
    if not states:
        return torch.zeros(len(rollouts), device=device)
    sites_t = torch.stack(sites).to(device)
    taken_t = torch.stack(taken).to(device)
    msites_t = torch.stack(msites).to(device)                     # (N, 16) bool
    mtoks_t = torch.stack(mtoks).to(device)                       # (N, 16) long
    mbans_t = torch.stack(mbans).to(device)                       # (N, 16) long
    temp = torch.tensor(temps, device=device)[:, None, None]      # (N, 1, 1)

    logits = model(torch.stack(states).to(device))                # (N, 16, 5)
    logp = torch.log_softmax(logits, dim=-1)
    p_mask = logp[..., MASK].exp()
    # remask decision: log p(MASK) if taken, log(1 - p(MASK)) otherwise
    keep = sites_t & ~taken_t
    per_state = (logp[..., MASK] * taken_t).sum(-1)
    per_state = per_state + (torch.log1p(-p_mask.clamp(max=1 - 1e-6)) * keep).sum(-1)
    # masked cells: 5-way choice (MASK = wait) at the rollout's temperature,
    # renormalized over the non-banned tokens exactly as at rollout time
    tlp = torch.log_softmax(_ban_logits(logits, mbans_t) / temp, dim=-1)  # (N, 16, 5)
    tok_lp = tlp.gather(-1, mtoks_t[..., None]).squeeze(-1)
    per_state = per_state + (tok_lp * msites_t).sum(-1)

    owner_t = torch.tensor(owner, device=device)
    return torch.zeros(len(rollouts), device=device).index_add(0, owner_t, per_state)
