#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beam gap 4096: одна перестановка за раз (без батчей).
beam_width=4096, depth=4096, gap-эвристика, checkpoint, сабмит на Kaggle.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import parse_permutation, moves_to_str, pancake_sort_moves  # noqa: E402
from src.heuristics.gpu_beam import beam_improve_or_baseline_h_gpu  # noqa: E402
from src.submission import log_experiment  # noqa: E402
import main as main_mod  # noqa: E402

PROGRESS_TIME_LIMIT_SEC = 5.0


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Beam gap 4096: одна перестановка за раз (single)."
    )
    parser.add_argument("--test", default="baseline/sample_submission.csv", help="CSV id, permutation.")
    parser.add_argument("--limit", type=int, default=None, help="Макс. строк из теста.")
    parser.add_argument("--out", default="submission_beam_gap_4096_single.csv", help="Выходной CSV.")
    parser.add_argument("--device", default=None, help="cuda / cpu.")
    parser.add_argument("--log-every", type=int, default=20, help="Лог каждые N строк.")
    args = parser.parse_args()

    test_path = Path(args.test)
    if not test_path.exists():
        print(f"ОШИБКА: не найден {test_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(test_path)
    if "id" not in df.columns or "permutation" not in df.columns:
        print(f"ОШИБКА: нужны колонки id, permutation", file=sys.stderr)
        sys.exit(1)
    if args.limit:
        df = df.head(args.limit).copy()

    ids = df["id"].astype(int).to_list()
    perms = [parse_permutation(p) for p in df["permutation"].to_list()]
    ns = [len(p) for p in perms]
    total = len(ids)
    out_path = Path(args.out)

    _log(f"[beam_gap_4096_single] Тест: {test_path} ({total} строк) | beam=4096 depth=4096")

    results: Dict[int, str] = {}
    if out_path.exists():
        try:
            ck = pd.read_csv(out_path)
            if "id" in ck.columns and "solution" in ck.columns:
                results = dict(zip(ck["id"].astype(int), ck["solution"].astype(str)))
                _log(f"[beam_gap_4096_single] Checkpoint: {len(results)} строк")
        except Exception as e:
            _log(f"[beam_gap_4096_single] Checkpoint не загружен: {e}")

    t0 = time.time()
    last_log_ts = t0
    n_to_indices: Dict[int, List[int]] = {}
    for idx, n_val in enumerate(ns):
        n_to_indices.setdefault(n_val, []).append(idx)
    n_order = sorted(n_to_indices.keys())

    def save_checkpoint() -> None:
        out_rows = [{"id": rid, "solution": results[rid]} for rid in ids if rid in results]
        if out_rows:
            pd.DataFrame(out_rows).to_csv(out_path, index=False)

    solved = len(results)
    for n_val in n_order:
        idx_list = n_to_indices[n_val]
        t_group_start = time.time()
        to_solve = [k for k in idx_list if ids[k] not in results]
        for k in idx_list:
            rid = ids[k]
            if rid in results:
                continue
            perm = perms[k]
            moves = beam_improve_or_baseline_h_gpu(
                perm,
                baseline_moves_fn=pancake_sort_moves,
                beam_width=4096,
                depth=4096,
                w=0.5,
                device=args.device,
                log=False,
            )
            results[rid] = moves_to_str(moves)
            solved += 1
            now = time.time()
            need_log = solved % max(1, args.log_every) == 0 or (now - last_log_ts) >= PROGRESS_TIME_LIMIT_SEC or solved == total
            if need_log or solved == total:
                elapsed = now - t0
                _log(f"[beam_gap_4096_single] {solved}/{total} | {solved / max(elapsed, 0.001):.2f} row/s")
                last_log_ts = now
        dt = time.time() - t_group_start
        if to_solve:
            _log(f"[beam_gap_4096_single] n={n_val} ({len(to_solve)} строк): {dt:.1f}s")
        save_checkpoint()

    if len(results) < total:
        _log(f"[beam_gap_4096_single] Неполный сабмит: {len(results)}/{total}")
    rows = [{"id": rid, "solution": results[rid]} for rid in ids if rid in results]
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    total_moves = out_df["solution"].fillna("").apply(
        lambda s: s.count(".") + 1 if isinstance(s, str) and s.strip() else 0
    ).sum()
    _log(f"[beam_gap_4096_single] Готово: {len(out_df)} строк → {out_path.name} | score={int(total_moves)}")

    log_experiment(
        script="run_beam_gap_4096_single",
        command="solve",
        method="beam_gap_4096_single",
        test_path=str(test_path),
        out_path=str(out_path),
        score=int(total_moves),
        n_rows=len(out_df),
    )
    competition = os.environ.get("KAGGLE_COMPETITION", main_mod._DEFAULT_KAGGLE_COMPETITION)
    _log(f"[beam_gap_4096_single] Сабмит: {out_path.name} competition={competition!r}")
    main_mod._do_kaggle_submit(str(out_path), competition, f"beam gap 4096 single (limit={args.limit or 'all'})")


if __name__ == "__main__":
    main()
