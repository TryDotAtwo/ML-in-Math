# -*- coding: utf-8 -*-
"""Эвристики качества состояния для pancake-задачи."""

from __future__ import annotations

from typing import List, Callable


def breakpoints2(state: List[int]) -> int:
    """Число «разрывов»: соседние элементы с разностью по модулю != 1.
    Дополнительно +1, если первый элемент не 0.
    """
    n = len(state)
    b = 0
    for i in range(n - 1):
        if abs(state[i] - state[i + 1]) != 1:
            b += 1
    if n > 0 and state[0] != 0:
        b += 1
    return b


def gap_h(state: List[int]) -> int:
    """Разрывы в последовательности с виртуальными границами -1 и n."""
    n = len(state)
    prev = -1
    gaps = 0
    for x in state:
        if abs(x - prev) != 1:
            gaps += 1
        prev = x
    if abs(n - prev) != 1:
        gaps += 1
    return gaps


def mix_h(state: List[int], alpha: float = 0.5) -> float:
    """Комбинация gap_h и breakpoints2 с весом alpha."""
    return gap_h(state) + alpha * breakpoints2(state)


def make_h(alpha: float) -> Callable[[List[int]], float]:
    """Фабрика эвристики: при alpha=0 только gap_h, иначе mix_h."""
    if alpha == 0.0:
        return lambda s: float(gap_h(s))
    return lambda s: float(mix_h(s, alpha=alpha))
