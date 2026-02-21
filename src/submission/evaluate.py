# -*- coding: utf-8 -*-
"""Сравнение сабмита с baseline по длине решений."""

from __future__ import annotations

import time
from typing import Dict, Optional, Callable, List, Iterable

import pandas as pd

from ..core.permutation import parse_permutation
from ..core.moves import moves_len


def evaluate_submission_vs_baseline(
    test_df: pd.DataFrame,
    submission_df: pd.DataFrame,
    *,
    baseline_moves_fn: Callable[[Iterable[int]], List[int]],
    log_every: int = 0,
    save_detailed_path: Optional[str] = None,
) -> Dict:
    """Сравнивает решение из submission_df с baseline на test_df.
    Ожидаются колонки: id, permutation (в test_df); id, solution (в submission_df).
    """
    t0 = time.time()
    sub_map = dict(zip(submission_df["id"].astype(int), submission_df["solution"].astype(str)))

    sum_base = 0
    sum_sub = 0
    improved = same = worse = 0
    total_gain_pos = 0
    max_gain = 0
    max_gain_id = None

    N = len(test_df)
    detailed_rows = [] if save_detailed_path else None

    for i, row in enumerate(test_df.itertuples(index=False), start=1):
        rid = int(row.id)
        perm = parse_permutation(row.permutation)

        base = baseline_moves_fn(perm)
        lb = len(base)

        sol = sub_map.get(rid, "")
        lz = moves_len(sol)

        sum_base += lb
        sum_sub += lz

        gain = lb - lz
        if gain > 0:
            improved += 1
            total_gain_pos += gain
            if gain > max_gain:
                max_gain = gain
                max_gain_id = rid
        elif gain == 0:
            same += 1
        else:
            worse += 1

        if detailed_rows is not None:
            detailed_rows.append({
                "id": rid,
                "n": len(perm),
                "base_len": lb,
                "sub_len": lz,
                "gain": gain,
            })

        if log_every and (i % log_every == 0 or i == 1 or i == N):
            print(
                f"[{i:4d}/{N}] base={lb:3d} sub={lz:3d} gain={gain:3d}  elapsed={time.time()-t0:7.1f}s",
                flush=True,
            )

    dt = time.time() - t0

    if save_detailed_path and detailed_rows:
        pd.DataFrame(detailed_rows).to_csv(save_detailed_path, index=False)

    return {
        "baseline_total": sum_base,
        "submission_total": sum_sub,
        "total_gain": (sum_base - sum_sub),
        "improved_cases": improved,
        "same_cases": same,
        "worse_cases": worse,
        "avg_gain_when_improved": (total_gain_pos / improved) if improved else 0.0,
        "max_gain": max_gain,
        "max_gain_id": max_gain_id,
        "time_sec": dt,
        "sec_per_sample": dt / max(1, N),
        "mean_baseline_len": sum_base / max(1, N),
        "mean_submission_len": sum_sub / max(1, N),
        "improved_frac": improved / max(1, N),
    }
