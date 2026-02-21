# -*- coding: utf-8 -*-
"""Классический жадный pancake sort — эталонный baseline O(n²) ходов."""

from __future__ import annotations

from typing import List, Iterable


def pancake_sort_moves(perm: Iterable[int]) -> List[int]:
    """Строит допустимую последовательность префиксных разворотов для перестановки.
    Гарантированно приводит к identity. Используется как baseline и верхняя граница.
    """
    a = list(perm)
    n = len(a)
    if n <= 1:
        return []

    pos = [0] * n
    for i, v in enumerate(a):
        pos[v] = i

    moves: List[int] = []

    def do_flip(k: int) -> None:
        if k <= 1:
            return
        i, j = 0, k - 1
        while i < j:
            vi, vj = a[i], a[j]
            a[i], a[j] = vj, vi
            pos[vi], pos[vj] = j, i
            i += 1
            j -= 1

    for target in range(n - 1, 0, -1):
        idx = pos[target]
        if idx == target:
            continue
        if idx != 0:
            do_flip(idx + 1)
            moves.append(idx + 1)
        do_flip(target + 1)
        moves.append(target + 1)

    return moves


def pancake_sort_path(perm: Iterable[int]) -> List[str]:
    """Классический pancake sort, возвращает список строк 'Rk' (как в блокноте)."""
    moves = pancake_sort_moves(perm)
    return [f"R{k}" for k in moves]
