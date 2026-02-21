# -*- coding: utf-8 -*-
"""Утилиты из блокнота: revers_perm, prob_step, извлечение ходов из результата солвера."""

from __future__ import annotations

from typing import List, Tuple, Any


def prob_step(perm: List[int]) -> int:
    """Эвристика «число шагов до цели» (нижняя оценка) для анализа качества перестановки."""
    perm_ = list(perm)
    n = len(perm_)
    k = 1 if perm_[-1] == n - 1 else 0
    if n >= 2 and sum(perm_[:2]) == 1:
        k -= 1
    while len(perm_) > 1:
        m = perm_.pop()
        k += 1 if (perm_[-1] - 1 == m or perm_[-1] + 1 == m) else 0
    return n - k


def revers_perm(perm) -> Tuple[List[int], dict]:
    """Обратная перестановка: perm[i] = j => reversed_perm[j] = i. Возвращает (reversed_perm, decode_dict)."""
    code_dict = {}
    decode_dict = {}
    for i, n in enumerate(perm):
        code_dict[n] = i
        decode_dict[i] = n
    reversed_perm = [code_dict[i] for i in range(len(perm))]
    return reversed_perm, decode_dict


def steps_from_solver_result(result) -> Tuple[List[int], int, int]:
    """Из возврата солвера (v3_1/v3_5/v4-стиль: (moves_or_pair, _, mlen, iter)) извлекает (steps, mlen, iter)."""
    if result is None or len(result) < 4:
        return [], 0, 0
    first, _, mlen, it = result[0], result[1], result[2], result[3]
    if isinstance(first, tuple) and len(first) >= 2:
        steps = list(first[0])  # (moves_tuple, stat) или (moves, stat, way)
    else:
        steps = list(first) if first is not None else []
    return steps, mlen, it
