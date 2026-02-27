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


def count_singletons(state: List[int]) -> int:
    """Число singletons — элементов с gap по обе стороны (Rokicki).
    Singleton легче «ориентировать» при flip → больше singletons = больше свободы.
    Используется как tie-breaking в beam: при равном h предпочесть больше singletons.
    O(N) по времени.
    """
    n = len(state)
    if n <= 1:
        return 0
    ext = [-1] + list(state) + [n]
    cnt = 0
    for i in range(1, len(ext) - 1):
        left_gap = abs(ext[i] - ext[i - 1]) != 1
        right_gap = abs(ext[i] - ext[i + 1]) != 1
        if left_gap and right_gap:
            cnt += 1
    return cnt


def _has_gap_decreasing_move(state: List[int]) -> bool:
    """Проверяет наличие хотя бы одного gap-decreasing хода за O(N).
    Ход M_i gap-decreasing ⟺ state[0] и state[i] «встанут» рядом
    с правильным соседом после flip. Достаточно проверить два потенциальных
    gap-resolving хода (Valenzano & Yang 2017, Section 6.1).
    """
    n = len(state)
    if n <= 1:
        return False
    ext = list(state) + [n]
    for i in range(1, n):
        top_val = state[0]
        below_val = ext[i + 1] if i + 1 <= n else n
        top_meets_below = abs(top_val - below_val) == 1
        gap_at_i = abs(state[i] - ext[i + 1]) != 1 if i < n else abs(state[i] - n) != 1
        if top_meets_below and gap_at_i:
            return True
    return False


def ld_h(state: List[int]) -> int:
    """Lock Detection heuristic (Valenzano & Yang 2017).
    LD(π) = gap_h(π) + 1  если состояние locked (нет gap-decreasing хода),
    LD(π) = gap_h(π)       иначе.
    Admissible, consistent, вычисляется за O(N).
    """
    g = gap_h(state)
    if g == 0:
        return 0
    if _has_gap_decreasing_move(state):
        return g
    return g + 1


def make_h(alpha: float) -> Callable[[List[int]], float]:
    """Фабрика эвристики: при alpha=0 только gap_h, иначе mix_h."""
    if alpha == 0.0:
        return lambda s: float(gap_h(s))
    return lambda s: float(mix_h(s, alpha=alpha))


def make_h_ld() -> Callable[[List[int]], float]:
    """Фабрика эвристики LD (lock detection) — улучшенная gap_h."""
    return lambda s: float(ld_h(s))


def make_h_singleton_tiebreak(base_h: Callable[[List[int]], float]) -> Callable[[List[int]], float]:
    """Обёртка: base_h - epsilon * count_singletons для tie-breaking в beam."""
    eps = 0.001
    return lambda s: base_h(s) - eps * count_singletons(s)
