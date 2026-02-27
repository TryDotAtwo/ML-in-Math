# -*- coding: utf-8 -*-
"""Обучение политики: behavior cloning по траекториям baseline, опционально policy gradient."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Callable, Optional

import torch
import torch.nn.functional as F

from ..core.baseline import pancake_sort_moves
from ..core.moves import apply_move_copy, is_solved
from .env import PancakeEnv
from .policy import PolicyNet


def random_perm(n: int, rng: random.Random) -> List[int]:
    perm = list(range(n))
    rng.shuffle(perm)
    return perm


def trajectories_from_baseline(
    n: int,
    num_trajectories: int,
    baseline_fn: Callable[[List[int]], List[int]],
    rng: random.Random,
    max_steps: Optional[int] = None,
) -> List[tuple]:
    """Список (state, action) пар по траекториям baseline. action = k (2..n)."""
    pairs: List[tuple] = []
    for _ in range(num_trajectories):
        perm = random_perm(n, rng)
        moves = baseline_fn(perm)
        if max_steps is not None and len(moves) > max_steps:
            continue
        state = perm
        for k in moves:
            pairs.append((list(state), k))
            state = apply_move_copy(state, k)
            if is_solved(state):
                break
    return pairs


def train_bc(
    n: int,
    num_trajectories: int = 5000,
    batch_size: int = 256,
    epochs: int = 30,
    lr: float = 1e-3,
    emb_dim: int = 32,
    hidden: int = 128,
    num_layers: int = 2,
    device: Optional[torch.device] = None,
    seed: int = 42,
    save_path: Optional[Path] = None,
    baseline_fn: Optional[Callable[[List[int]], List[int]]] = None,
) -> PolicyNet:
    """Behavior cloning: обучает PolicyNet(n) предсказывать ход baseline по состоянию."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if baseline_fn is None:
        baseline_fn = pancake_sort_moves
    rng = random.Random(seed)
    torch.manual_seed(seed)

    pairs = trajectories_from_baseline(n, num_trajectories, baseline_fn, rng)
    if not pairs:
        raise RuntimeError(f"No trajectories for n={n}")

    model = PolicyNet(n, emb_dim=emb_dim, hidden=hidden, num_layers=num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        total_loss = 0.0
        num_batches = 0
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            states = torch.tensor([s for s, _ in batch], dtype=torch.long, device=device)
            actions = torch.tensor([a - 2 for _, a in batch], dtype=torch.long, device=device)
            logits = model(states)
            loss = F.cross_entropy(logits, actions)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            num_batches += 1
        avg = total_loss / max(1, num_batches)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  n={n} epoch={epoch+1}/{epochs} loss={avg:.4f}", flush=True)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "n": n,
                "emb_dim": emb_dim,
                "hidden": hidden,
                "num_layers": num_layers,
                "state_dict": model.state_dict(),
            },
            save_path,
        )
    return model


def train_bc_all_n(
    n_list: List[int],
    num_trajectories_per_n: int = 5000,
    **kwargs,
) -> None:
    """Обучает и сохраняет политику для каждого n из n_list."""
    save_dir = kwargs.pop("save_dir", Path("runs/rl_models"))
    save_dir = Path(save_dir)
    for n in n_list:
        save_path = save_dir / f"policy_n_{n}.pt"
        train_bc(n, num_trajectories=num_trajectories_per_n, save_path=save_path, **kwargs)


def train_pg_epochs(
    model: PolicyNet,
    n: int,
    num_rollouts: int = 500,
    epochs: int = 10,
    lr: float = 1e-4,
    device: Optional[torch.device] = None,
    seed: int = 42,
    baseline_fn: Optional[Callable[[List[int]], List[int]]] = None,
) -> PolicyNet:
    """Policy gradient (REINFORCE): максимизировать return = -len(trajectory). Улучшает BC-политику."""
    if device is None:
        device = next(model.parameters()).device
    if baseline_fn is None:
        baseline_fn = pancake_sort_moves
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    env = PancakeEnv(list(range(n)))

    for epoch in range(epochs):
        model.train()
        total_reward = 0.0
        total_loss = 0.0
        count = 0
        for _ in range(num_rollouts):
            perm = random_perm(n, rng)
            env.reset(perm)
            log_probs = []
            rewards = []
            state = perm
            while not is_solved(state):
                x = torch.tensor([state], dtype=torch.long, device=device)
                logits = model(x)[0]
                dist = torch.distributions.Categorical(logits=logits)
                k_idx = dist.sample()
                k = 2 + k_idx.item()
                log_probs.append(dist.log_prob(k_idx))
                state, r, done, _ = env.step(k)
                rewards.append(r)
            returns = []
            R = 0.0
            for r in reversed(rewards):
                R = r + 0.99 * R
                returns.insert(0, R)
            loss = -sum(lp * R for lp, R in zip(log_probs, returns))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_reward += sum(rewards)
            total_loss += loss.item()
            count += 1
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  PG n={n} epoch={epoch+1}/{epochs} mean_return={total_reward/max(1,count):.1f} loss={total_loss/max(1,count):.4f}", flush=True)
    return model
