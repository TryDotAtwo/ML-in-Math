# -*- coding: utf-8 -*-
"""Обработка одной строки теста: прогон через солвер, формат для сабмита."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from ..core.permutation import parse_permutation
from ..core.moves import moves_to_str
from ..notebook_search.utils import revers_perm, steps_from_solver_result


def process_row(
    row: dict,
    func: Callable,
    treshold: int = 3,
    save: bool = False,
    from_target: bool = False,
    save_dir: Optional[Path] = None,
) -> dict:
    """Прогон одной строки: perm из row, вызов func(perm, treshold).
    func должен возвращать (moves_or_pair, _, mlen, iter) в стиле v3_1/v3_5/v4.
    Если from_target — применяется revers_perm к perm и ходы разворачиваются.
    """
    perm = parse_permutation(row["permutation"])

    if from_target:
        perm, _ = revers_perm(perm)

    result = func(perm, treshold)
    steps, mlen, it = steps_from_solver_result(result)

    if from_target:
        steps = steps[::-1]

    path_str = moves_to_str(steps) if steps else "UNSOLVED"

    if save and save_dir is not None:
        id_ = row["id"]
        n = int(row.get("n", len(perm)))
        save_dir.mkdir(parents=True, exist_ok=True)
        path_str = moves_to_str(steps) if steps else "UNSOLVED"
        with open(save_dir / f"n_{n}.txt", mode="a", encoding="utf-8") as f:
            f.write(f"{id_} - {path_str}\n")

    return {
        "id": row["id"],
        "permutation": row["permutation"],
        "solution": path_str,
        "score": len(steps),
        "n": int(row.get("n", len(perm))),
        "mlen": mlen,
        "iter": it,
    }
