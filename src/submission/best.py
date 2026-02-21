# -*- coding: utf-8 -*-
"""Выбор лучшего сабмита из текущего и сохранённого best."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def best_solution(
    submission_df: pd.DataFrame,
    best_df: Optional[pd.DataFrame] = None,
    best_path: Optional[str] = None,
    safe: bool = False,
):
    """Сравнивает submission_df с best_df по score; обновляет best там, где submission лучше.
    Если best_df не передан и задан best_path — загружает из CSV.
    Если safe=True и были улучшения — сохраняет best_df в best_path (если задан).
    Возвращает (best_df, stats_dict) где stats = {'best': N, 'same': N, 'worse': N}.
    """
    if best_df is None and best_path is not None:
        best_df = pd.read_csv(best_path)
    elif best_df is None:
        best_df = submission_df.copy()

    if "score" not in best_df.columns and "solution" in best_df.columns:
        best_df["score"] = best_df["solution"].fillna("").apply(lambda s: s.count(".") + 1 if isinstance(s, str) and s.strip() else 0)
    sub = submission_df.copy()
    if "score" not in sub.columns and "solution" in sub.columns:
        sub["score"] = sub["solution"].fillna("").apply(lambda s: s.count(".") + 1 if isinstance(s, str) and s.strip() else 0)

    best_df = best_df.set_index("id")
    submission_df = sub.set_index("id")

    common_idx = best_df.index.intersection(submission_df.index)

    same_score_idx = []
    best_score_idx = []

    for idx in common_idx:
        if best_df.loc[idx, "score"] == submission_df.loc[idx, "score"]:
            same_score_idx.append(idx)
        if best_df.loc[idx, "score"] > submission_df.loc[idx, "score"]:
            best_score_idx.append(idx)
            best_df.loc[idx, ["solution", "score"]] = submission_df.loc[idx, ["solution", "score"]]

    if safe and best_score_idx and best_path:
        best_df.reset_index().to_csv(best_path, index=False)
        print("Best submission updated.")

    return best_df.reset_index().sort_values("id"), {
        "best": len(best_score_idx),
        "same": len(same_score_idx),
        "worse": len(common_idx) - len(best_score_idx) - len(same_score_idx),
    }
