# -*- coding: utf-8 -*-
"""Проверка корректности решений в DataFrame (id, permutation, solution)."""

from __future__ import annotations

from typing import List

import pandas as pd

from ..core.permutation import parse_permutation
from ..core.moves import solution_to_moves, apply_moves


def check_steps(df: pd.DataFrame) -> List[int]:
    """Для каждой строки проверяет, что solution приводит permutation к identity.
    Возвращает список id с неверным решением; при неверном решении печатает сообщение.
    """
    wrong_ids = []
    for row in df.to_dict("records"):
        perm = parse_permutation(row["permutation"])
        sol_str = row.get("solution", "")
        steps = solution_to_moves(sol_str)
        final = apply_moves(perm, steps)
        if final != list(range(len(perm))):
            wrong_ids.append(row["id"])
            print(f"Wrong solution for id {row['id']}")
    return wrong_ids
