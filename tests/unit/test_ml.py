# -*- coding: utf-8 -*-
"""Тесты для src/ml: env, policy, inference fallback. Требуют torch."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from src.core import parse_permutation, apply_moves, is_solved, pancake_sort_moves
from src.ml.env import PancakeEnv
from src.ml.policy import PolicyNet, policy_forward
from src.ml.inference import solve_with_rl_or_baseline, rollout_policy, load_policy_for_n


def test_env_step():
    env = PancakeEnv([1, 0, 2])
    next_s, r, done, info = env.step(2)
    assert r == -1.0
    assert not done
    assert next_s == [0, 1, 2]
    next_s, r, done, info = env.step(2)
    assert done
    assert is_solved(next_s)


def test_policy_net_forward():
    import torch
    n = 5
    model = PolicyNet(n)
    state = [2, 1, 0, 3, 4]
    x = torch.tensor([state], dtype=torch.long)
    logits = model(x)
    assert logits.shape == (1, n - 1)


def test_solve_with_rl_or_baseline_fallback():
    """Без обученной модели должен использоваться baseline."""
    perm = [1, 0, 2]
    moves = solve_with_rl_or_baseline(perm, "runs/rl_models_nonexistent")
    assert apply_moves(perm, moves) == list(range(3))
    assert moves == pancake_sort_moves(perm)


def test_load_policy_for_n_missing():
    assert load_policy_for_n(7, "runs/nonexistent_dir") is None
