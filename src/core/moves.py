# -*- coding: utf-8 -*-
"""Ходы префиксного разворота: применение, формат строки, проверка решения."""

from __future__ import annotations

from typing import List, Any


def moves_to_str(moves: List[int]) -> str:
    """Список ходов в строку формата сабмита: 'R2.R5.R3'."""
    return ".".join(f"R{k}" for k in moves)


def moves_len(sol: Any) -> int:
    """Число ходов в решении. sol — строка 'R2.R5' или список ходов."""
    if sol is None:
        return 0
    if isinstance(sol, (list, tuple)):
        return len(sol)
    s = str(sol).strip()
    if s == "" or (isinstance(sol, float) and sol != sol):  # NaN
        return 0
    return s.count(".") + 1


def solution_to_moves(sol: str | None) -> List[int]:
    """Строку решения 'R2.R5.R3' в list[int] ходов [2, 5, 3]."""
    if sol is None or not str(sol).strip():
        return []
    s = str(sol).strip()
    parts = s.split(".")
    out = []
    for p in parts:
        p = p.strip()
        if p.startswith("R"):
            try:
                out.append(int(p[1:].strip()))
            except ValueError:
                continue
        elif p.isdigit():
            out.append(int(p))
    return out


def apply_move_copy(state: List[int], k: int) -> List[int]:
    """Разворот первых k элементов. Не меняет state, возвращает новый список."""
    nxt = state[:]
    nxt[:k] = reversed(nxt[:k])
    return nxt


def apply_moves(perm: List[int], moves: List[int]) -> List[int]:
    """Применить последовательность ходов к перестановке (копия)."""
    a = perm[:]
    for k in moves:
        a[:k] = reversed(a[:k])
    return a


def is_solved(state: List[int]) -> bool:
    """Проверка: state == [0, 1, ..., n-1]."""
    return all(v == i for i, v in enumerate(state))
