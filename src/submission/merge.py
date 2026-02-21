# -*- coding: utf-8 -*-
"""Сохранение прогресса и объединение сабмитов (base + partial)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

import pandas as pd

from ..core.moves import moves_len


def save_progress(progress_map: Dict[int, str], path: str) -> None:
    """Сохраняет progress_map (id -> solution string) в CSV."""
    df = pd.DataFrame(
        list(progress_map.items()), columns=["id", "solution"]
    ).sort_values("id")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


def merge_submissions_with_partials(
    *,
    base_paths: List[str],
    partial_paths: List[str],
    out_path: str = "submission_final.csv",
    save_source_column: bool = True,
    tie_break: str = "keep_base",
) -> pd.DataFrame:
    """Объединяет базовые сабмиты и поверх применяет partial (замена только при строгом улучшении).
    base_paths[0] задаёт список id; остальные base должны совпадать. partial могут быть неполными.
    """
    base_subs: List[pd.DataFrame] = []
    for p in base_paths:
        df = pd.read_csv(p)
        assert {"id", "solution"}.issubset(df.columns), f"{p} must have columns: id, solution"
        df = df[["id", "solution"]].copy()
        df["id"] = df["id"].astype(int)
        df = df.sort_values("id").reset_index(drop=True)
        df["len"] = df["solution"].map(moves_len)
        df["source"] = "baseline"
        base_subs.append(df)

    base_ids = base_subs[0]["id"].values
    for i, df in enumerate(base_subs[1:], start=1):
        if len(df) != len(base_subs[0]) or not (df["id"].values == base_ids).all():
            raise ValueError(f"ID mismatch between {base_paths[0]} and {base_paths[i]}")

    best = base_subs[0][["id", "solution", "len", "source"]].copy()
    for df in base_subs[1:]:
        better = df["len"].values < best["len"].values
        best.loc[better, "solution"] = df.loc[better, "solution"].values
        best.loc[better, "len"] = df.loc[better, "len"].values
        best.loc[better, "source"] = df.loc[better, "source"].values

    id_to_idx: Dict[int, int] = {int(pid): i for i, pid in enumerate(best["id"].values)}

    for p in partial_paths:
        if not Path(p).exists():
            print(f"[WARN] partial not found, skipped: {p}")
            continue

        part = pd.read_csv(p)
        assert {"id", "solution"}.issubset(part.columns), f"{p} must have columns: id, solution"
        part = part[["id", "solution"]].copy()
        part["id"] = part["id"].astype(int)
        part["len"] = part["solution"].map(moves_len)

        stem = Path(p).stem
        src_tag = stem.replace("submission_", "")

        for _, r in part.iterrows():
            pid, sol, l = int(r["id"]), r["solution"], int(r["len"])
            idx = id_to_idx.get(pid)
            if idx is None:
                continue
            cur_len = int(best.at[idx, "len"])
            if l < cur_len or (
                tie_break == "prefer_partial" and l == cur_len and sol != best.at[idx, "solution"]
            ):
                best.at[idx, "solution"] = sol
                best.at[idx, "len"] = l
                best.at[idx, "source"] = src_tag

    out_df = best[["id", "solution"]].copy()
    if save_source_column:
        out_df["source"] = best["source"].copy()

    out_df.to_csv(out_path, index=False)

    total_moves = int(best["len"].sum())
    print("\n=== MERGE SUMMARY ===")
    print("Output:", out_path)
    print("Rows:", len(out_df))
    print("Total moves (score):", total_moves)
    if save_source_column:
        print("\nFinal winners by source tag (top):")
        print(out_df["source"].value_counts().head(20).to_string())
    print("\nSaved:", out_path)
    return out_df
