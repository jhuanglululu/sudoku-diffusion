"""GRPO on sampler trajectories, initialized from an SFT checkpoint.

Per step: sample `puzzles_per_batch` puzzles from train-orbit solutions
(clue counts from clue_counts; 0 = empty board), roll out `group_size`
stochastic trajectories per puzzle, reward each final board:

    reward = validity_score + (solved ? solved_bonus * (1 + alpha * (1 - steps_used/max_steps)) : 0)

so solving in fewer sampler steps scores strictly higher ("thinking budget").
Advantages are group-normalized; the loss is the clipped PPO-style objective
with old logprobs recorded at rollout time.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from tqdm import tqdm

from .data import SPLIT_SEED, make_puzzle, orbit_split, solved, validity_score
from .model import SudokuDenoiser, get_device
from .runs import Record, load_checkpoint, save_checkpoint, seed_all
from .sampler import action_logprob, sample
from .variations import MODELS, TRAININGS, GRPOConfig


def reward(final_board: np.ndarray, puzzle: np.ndarray, steps_used: int, cfg: GRPOConfig) -> tuple[float, bool]:
    ok = solved(final_board, puzzle)
    r = validity_score(final_board)
    if ok:
        r += cfg.solved_bonus * (1 + cfg.efficiency_alpha * (1 - steps_used / cfg.max_sample_steps))
    return r, ok


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """rewards: (P, G) -> normalized advantages per group."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + 1e-4)


def sample_puzzles(train_sols: np.ndarray, cfg: GRPOConfig, rng: np.random.Generator) -> np.ndarray:
    out = []
    attempts, max_attempts = 0, cfg.puzzles_per_batch * 200
    while len(out) < cfg.puzzles_per_batch:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"made {len(out)}/{cfg.puzzles_per_batch} puzzles in {max_attempts} attempts; "
                f"clue_counts {cfg.clue_counts} may not admit unique puzzles"
            )
        n_clues = int(rng.choice(cfg.clue_counts))
        if n_clues == 0:
            out.append(np.zeros(16, dtype=train_sols.dtype))
            continue
        sol = train_sols[rng.integers(len(train_sols))]
        puz = make_puzzle(sol, n_clues, rng)
        if puz is not None:
            out.append(puz)
    return np.stack(out)


def train_grpo(model_name: str, training_name: str, seed: int) -> None:
    mcfg, cfg = MODELS[model_name], TRAININGS[training_name]
    if not isinstance(cfg, GRPOConfig):
        raise ValueError(f"training '{training_name}' is kind={cfg.kind!r}, expected grpo")
    seed_all(seed)
    device = get_device()
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    train_sols, _ = orbit_split(np.random.default_rng(SPLIT_SEED))
    model = SudokuDenoiser(mcfg).to(device)
    meta = load_checkpoint(model, model_name, cfg.init_from_training, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, cfg.warmup_steps))
    )
    rec = Record(model_name, training_name, seed, {**mcfg.model_dump(), **cfg.model_dump(), "init_step": meta.get("step")})

    # The whole step runs in eval mode: rollouts, old and new logprobs must all
    # score the same policy, or dropout > 0 would bias the PPO ratio.
    model.eval()
    P, G = cfg.puzzles_per_batch, cfg.group_size
    best_solve = -1.0
    bar = tqdm(range(cfg.steps), desc=f"{model_name}/{training_name} e0")
    t0 = time.time()
    for step in bar:
        puzzles = sample_puzzles(train_sols, cfg, rng)
        batch = torch.from_numpy(np.repeat(puzzles, G, axis=0)).long().to(device)  # (P*G, 16)
        _, steps_used, rollouts, _ = sample(
            model, batch, cfg.max_sample_steps, cfg.commit_frac,
            temperature=cfg.temperature, generator=gen, record=True,
            commit_threshold=cfg.commit_threshold,
        )
        rewards = torch.zeros(P * G)
        solves = 0
        for i, r in enumerate(rollouts):
            rw, ok = reward(r.final_board.cpu().numpy(), puzzles[i // G], r.steps_used, cfg)
            rewards[i] = rw
            solves += ok
        adv = group_advantages(rewards.view(P, G)).view(-1).to(device)
        new_lp = action_logprob(model, rollouts)
        # single gradient step per rollout batch: the rollout policy IS the
        # current policy, so old_lp == new_lp — skip its forward pass. If inner
        # PPO epochs are ever added, compute old_lp before the first update.
        old_lp = new_lp.detach()
        ratio = torch.exp(new_lp - old_lp)
        clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
        loss = -torch.min(ratio * adv, clipped * adv).mean()
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        sched.step()

        solve_rate = solves / (P * G)
        mean_steps = float(steps_used.float().mean())
        if (step + 1) % cfg.log_every == 0:
            rec.write(
                type="step", step=step + 1, loss=round(loss.item(), 4),
                reward=round(rewards.mean().item(), 4), solve_rate=round(solve_rate, 4),
                mean_steps=round(mean_steps, 2), grad_norm=round(grad_norm, 4),
                lr=sched.get_last_lr()[0], sec_per_step=round((time.time() - t0) / (step + 1), 3),
            )
            bar.set_postfix(reward=f"{rewards.mean():.3f}", solve=f"{solve_rate:.2f}", steps=f"{mean_steps:.1f}")
        if (step + 1) % cfg.val_every == 0 or step + 1 == cfg.steps:
            el = int(time.time() - t0)
            tqdm.write(
                f"e0 | step {step + 1:>5}/{cfg.steps} | {el // 60:02d}:{el % 60:02d} | "
                f"reward {rewards.mean():6.3f} | solve {solve_rate:5.2f} | steps {mean_steps:5.2f}"
            )
            is_best = solve_rate > best_solve
            best_solve = max(best_solve, solve_rate)
            save_checkpoint(model, model_name, training_name, seed, step + 1, {"solve_rate": solve_rate}, best=is_best)
