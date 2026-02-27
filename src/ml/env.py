# -*- coding: utf-8 -*-
"""MDP-окружение для pancake sorting: состояние = перестановка, действие = k (разворот префикса)."""

from __future__ import annotations

from typing import List, Tuple, Any

from ..core.moves import apply_move_copy, is_solved


class PancakeEnv:
    """Один эпизод: начальная перестановка, шаги по действиям k=2..n, награда -1 за шаг."""

    def __init__(self, perm: List[int]):
        self.n = len(perm)
        self._state = list(perm)
        self._done = False
        self._steps = 0

    def reset(self, perm: List[int] | None = None) -> List[int]:
        if perm is not None:
            self._state = list(perm)
        else:
            self._state = list(range(self.n))
        self._done = False
        self._steps = 0
        return self._state.copy()

    @property
    def state(self) -> List[int]:
        return self._state.copy()

    def step(self, k: int) -> Tuple[List[int], float, bool, dict]:
        """k in [2, n]. Возвращает (next_state, reward, done, info)."""
        if self._done:
            return self._state.copy(), 0.0, True, {}
        if k < 2 or k > self.n:
            raise ValueError(f"Action k must be in [2, {self.n}], got {k}")
        self._state = apply_move_copy(self._state, k)
        self._steps += 1
        done = is_solved(self._state)
        self._done = done
        reward = -1.0
        return self._state.copy(), reward, done, {"steps": self._steps}

    def valid_actions(self) -> List[int]:
        """Допустимые действия: 2..n."""
        return list(range(2, self.n + 1))
