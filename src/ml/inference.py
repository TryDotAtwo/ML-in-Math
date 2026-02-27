# -*- coding: utf-8 -*-
"""Инференс: rollout политики до решения, fallback на baseline."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Callable

import torch

from ..core.moves import apply_move_copy, is_solved, moves_to_str
from ..core.baseline import pancake_sort_moves
from .policy import PolicyNet, policy_forward


def load_policy_for_n(
    n: int,
    path_or_dir: Path | str,
    device: Optional[torch.device] | str = None,
) -> Optional[PolicyNet]:
    """Загружает модель для данного n. path_or_dir — файл policy_n_{n}.pt или каталог с такими файлами."""
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / f"policy_n_{n}.pt"
    if not path.exists():
        return None
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model_n = ckpt.get("n", n)
    if model_n != n:
        return None
    model = PolicyNet(
        n,
        emb_dim=ckpt.get("emb_dim", 32),
        hidden=ckpt.get("hidden", 128),
        num_layers=ckpt.get("num_layers", 2),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def rollout_policy(
    perm: List[int],
    model: PolicyNet,
    device: torch.device,
    greedy: bool = True,
    max_steps: Optional[int] = None,
) -> List[int]:
    """Строит решение жадным rollout политики. Возвращает list[int] ходов."""
    state = list(perm)
    n = len(state)
    moves: List[int] = []
    limit = max_steps if max_steps is not None else 2 * n
    while not is_solved(state) and len(moves) < limit:
        k = policy_forward(model, state, device, greedy=greedy)
        moves.append(k)
        state = apply_move_copy(state, k)
    return moves


def solve_with_rl_or_baseline(
    perm: List[int],
    model_dir: Path | str,
    device: Optional[torch.device] | str = None,
    baseline_fn: Optional[Callable[[List[int]], List[int]]] = None,
) -> List[int]:
    """Решение: если есть обученная политика для n — rollout, иначе baseline."""
    n = len(perm)
    if baseline_fn is None:
        baseline_fn = pancake_sort_moves
    model = load_policy_for_n(n, model_dir, device)
    if model is None:
        return baseline_fn(perm)
    if device is None:
        device = next(model.parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)
    return rollout_policy(perm, model, device, greedy=True)
