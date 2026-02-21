# -*- coding: utf-8 -*-
"""Сводка по best_df по размерам n (score, prob_step, potential)."""

from __future__ import annotations

import pandas as pd

from ..core.permutation import parse_permutation
from ..notebook_search.utils import prob_step


def compare(best_df: pd.DataFrame, n_list=None) -> None:
    """Печатает по каждому n из n_list сводку: sum n, score, prob_step, potential (score - prob_step).
    Ожидает колонки n, score; при наличии prob_step использует их, иначе считает по permutation.
    """
    if n_list is None:
        n_list = [5, 12, 15, 16, 20, 25, 30, 35, 40, 45, 50, 75, 100]

    if "prob_step" not in best_df.columns and "permutation" in best_df.columns:
        best_df = best_df.copy()
        best_df["prob_step"] = best_df["permutation"].apply(
            lambda x: prob_step(parse_permutation(x))
        )

    for n in n_list:
        pos = best_df["n"] == n
        if not pos.any():
            continue
        score = best_df.loc[pos, "score"].sum()
        n_sum = best_df.loc[pos, "n"].sum()
        pstep = best_df.loc[pos, "prob_step"].sum()
        print(f"n: {n} | sum n: {n_sum} | score: {score} | prob step: {pstep} | potential: {score - pstep}")
    print()
    print(f"sum n: {best_df['n'].sum()} | score: {best_df['score'].sum()} | prob step: {best_df['prob_step'].sum()}")
