"""SFT: denoising on corrupted solutions + symmetry-consistency loss.

Corruption of a solution grid, per example:
- a random fraction of cells (uniform in mask_frac_range) is set to MASK
  (fraction 1.0 = fully empty board);
- of the remaining cells, a random fraction (wrong_frac_range) is replaced by
  a *wrong* digit.
Targets: masked cell -> correct digit; wrong cell -> MASK (teaches remasking);
untouched cell -> its own value (teaches keeping correct cells filled).

Consistency: a random geometric symmetry g per batch; loss includes symmetric
KL between logits(x) and g^-1(logits(g(x))).
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import MASK, orbit_split, symmetry_group
from .model import SudokuDenoiser, get_device
from .runs import Record, save_checkpoint, seed_all
from .variations import MODELS, TRAININGS

IGNORE = -100


def corrupt_batch(
    solutions: np.ndarray, cfg, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a corrupted batch. Returns (inputs, targets), both (B, 16)."""
    B = cfg.batch_size
    sols = solutions[rng.integers(len(solutions), size=B)]
    inputs = torch.from_numpy(sols).long()
    targets = torch.full_like(inputs, IGNORE)

    mask_frac = rng.uniform(*cfg.mask_frac_range, size=B)
    wrong_frac = rng.uniform(*cfg.wrong_frac_range, size=B)
    u = rng.random((B, 16))
    masked = torch.from_numpy(u < mask_frac[:, None])
    wrong = torch.from_numpy((u >= mask_frac[:, None]) & (u < (mask_frac + (1 - mask_frac) * wrong_frac)[:, None]))

    # wrong digit = correct + offset in {1,2,3} mod 4, mapped back to 1..4
    offs = torch.from_numpy(rng.integers(1, 4, size=(B, 16)))
    wrong_vals = ((inputs - 1 + offs) % 4) + 1

    # untouched correct cells: target = keep their own value — without this the
    # model has no signal against remasking correct cells and oscillates forever
    targets = inputs.clone()
    targets[masked] = inputs[masked]
    inputs[masked] = MASK
    targets[wrong] = MASK
    inputs[wrong] = wrong_vals[wrong]
    return inputs, targets


def sft_losses(model, inputs, targets, group: torch.Tensor, cfg, gen: torch.Generator):
    logits = model(inputs)
    ce = F.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1), ignore_index=IGNORE)

    g = group[torch.randint(len(group), (1,), generator=gen).item()]
    inv = torch.empty_like(g)
    inv[g] = torch.arange(16, device=g.device)
    logits_t = model(inputs[:, g])[:, inv]
    p, q = F.log_softmax(logits, -1), F.log_softmax(logits_t, -1)
    cons = 0.5 * (
        F.kl_div(q, p, log_target=True, reduction="batchmean")
        + F.kl_div(p, q, log_target=True, reduction="batchmean")
    )
    return ce, cons


def train_sft(model_name: str, training_name: str, seed: int) -> None:
    mcfg, cfg = MODELS[model_name], TRAININGS[training_name]
    assert cfg.kind == "sft"
    seed_all(seed)
    device = get_device()
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)

    train_sols, eval_sols = orbit_split(np.random.default_rng(12345))
    group = torch.from_numpy(symmetry_group()).long()
    model = SudokuDenoiser(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, cfg.warmup_steps))
    )
    rec = Record(model_name, training_name, seed, {**mcfg.model_dump(), **cfg.model_dump()})
    val_inputs, val_targets = corrupt_batch(eval_sols, cfg, np.random.default_rng(999))
    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

    bar = tqdm(range(cfg.steps), desc=f"{model_name}/{training_name} e0")
    t0 = time.time()
    for step in bar:
        inputs, targets = corrupt_batch(train_sols, cfg, rng)
        ce, cons = sft_losses(model, inputs.to(device), targets.to(device), group.to(device), cfg, gen)
        loss = ce + cfg.consistency_weight * cons
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        sched.step()

        if (step + 1) % cfg.log_every == 0:
            rec.write(
                type="step", step=step + 1, loss=round(loss.item(), 4), ce=round(ce.item(), 4),
                consistency=round(cons.item(), 4), lr=sched.get_last_lr()[0],
                grad_norm=round(grad_norm, 4), sec_per_step=round((time.time() - t0) / (step + 1), 4),
            )
            bar.set_postfix(loss=f"{loss.item():.3f}")
        if (step + 1) % cfg.val_every == 0 or step + 1 == cfg.steps:
            model.eval()
            with torch.no_grad():
                vce, vcons = sft_losses(model, val_inputs, val_targets, group.to(device), cfg, gen)
                val = (vce + cfg.consistency_weight * vcons).item()
            model.train()
            rec.write(type="eval", step=step + 1, val_loss=round(val, 4))
            el = int(time.time() - t0)
            diff = val - loss.item()
            tqdm.write(
                f"e0 | step {step + 1:>5}/{cfg.steps} | {el // 60:02d}:{el % 60:02d} | "
                f"loss {loss.item():6.3f} | val {val:6.3f} | diff {diff:+6.3f}"
            )
            bar.set_postfix(loss=f"{loss.item():.3f}", val=f"{val:.3f}")
            save_checkpoint(model, model_name, training_name, seed, step + 1, {"val_loss": val})
