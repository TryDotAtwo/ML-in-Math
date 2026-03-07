#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Запуск solve в специальном режиме:
- baseline + beam search с gap-эвристикой,
- ширина луча 2^12=4096 и глубина 4096,
- автоматический сабмит на Kaggle.

Обычно вызывать через корневой скрипт:

  python run_beam_gap_4096.py --limit 100 --out submission_beam_gap_4096.csv

Логика прогресса и формат логов сделаны по аналогии с scripts.runners.run_research:
печатаются таймстемпы, количество решённых строк, скорость и память.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd

# Корень проекта (ML in Math)
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
        description=(
            "Baseline + beam search с gap-эвристикой, beam_width=4096, depth=4096, "
            "с автоматическим сабмитом на Kaggle."
        )
    )
    parser.add_argument(
        "--test",
        default="baseline/sample_submission.csv",
        help="CSV с колонками id, permutation (по умолчанию baseline/sample_submission.csv).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Максимальное число строк из теста (по умолчанию все).",
    )
    parser.add_argument(
        "--out",
        default="submission2_beam_gap_4096.csv",
        help="Имя выходного CSV (id, solution).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device (например 'cuda' или 'cpu'; по умолчанию cuda, если доступна).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
        help="Как часто логировать прогресс по числу строк (по умолчанию каждые 20).",
    )
    args = parser.parse_args()

    test_path = Path(args.test)
    if not test_path.exists():
        print(f"ОШИБКА: файл теста не найден: {test_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(test_path)
    if "id" not in df.columns or "permutation" not in df.columns:
        print(f"ОШИБКА: {test_path} должен содержать колонки id, permutation", file=sys.stderr)
        sys.exit(1)

    if args.limit:
        df = df.head(args.limit).copy()

    # Предварительно распарсим перестановки один раз, чтобы в горячем цикле не делать парсинг строк.
    ids = df["id"].astype(int).to_list()
    perms = [parse_permutation(p) for p in df["permutation"].to_list()]
    ns = [len(p) for p in perms]

    total = len(ids)
    out_path = Path(args.out)

    _log(
        f"[beam_gap_4096] Тест: {test_path} ({total} строк) | "
        f"beam_width=4096 depth=4096 alpha=0.0 w=0.5"
    )

    # Результаты: id -> solution; при наличии checkpoint подгружаем и продолжаем.
    results: Dict[int, str] = {}
    if out_path.exists():
        try:
            ck = pd.read_csv(out_path)
            if "id" in ck.columns and "solution" in ck.columns:
                results = dict(zip(ck["id"].astype(int), ck["solution"].astype(str)))
                _log(f"[beam_gap_4096] Загружен checkpoint: {len(results)} строк из {out_path.name}")
        except Exception as e:
            _log(f"[beam_gap_4096] Не удалось загрузить checkpoint: {e}")

    t0 = time.time()
    last_log_ts = t0

    # Одна перестановка за раз (без батчей).
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
            need_log_by_count = solved % max(1, args.log_every) == 0
            need_log_by_time = now - last_log_ts >= PROGRESS_TIME_LIMIT_SEC
            if need_log_by_count or need_log_by_time or solved == total:
                elapsed = now - t0
                speed = solved / max(elapsed, 0.001)
                _log(f"[beam_gap_4096] {solved}/{total} решено | {speed:.2f} row/s")
                last_log_ts = now
        dt = time.time() - t_group_start
        if to_solve:
            _log(f"[beam_gap_4096] n={n_val} ({len(to_solve)} строк): {dt:.1f}s, {len(to_solve) / max(dt, 0.001):.1f} row/s")
        save_checkpoint()

    if len(results) < total:
        _log(f"[beam_gap_4096] Внимание: решено {len(results)} из {total}, сабмит неполный (можно перезапустить для дозаполнения).")
    rows = [{"id": rid, "solution": results[rid]} for rid in ids if rid in results]
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)

    total_moves = out_df["solution"].fillna("").apply(
        lambda s: s.count(".") + 1 if isinstance(s, str) and s.strip() else 0
    ).sum()
    _log(
        f"[beam_gap_4096] Готово: {len(out_df)} строк → {out_path.name} | "
        f"total moves (score) = {int(total_moves)}"
    )

    # Лог эксперимента в общий JSONL лог.
    log_experiment(
        script="run_beam_gap_4096",
        command="solve",
        method="beam_gap_4096",
        test_path=str(test_path),
        out_path=str(out_path),
        score=int(total_moves),
        n_rows=len(out_df),
    )

    # Автоматический сабмит на Kaggle.
    competition = os.environ.get("KAGGLE_COMPETITION", main_mod._DEFAULT_KAGGLE_COMPETITION)
    _log(
        f"[beam_gap_4096] Сабмит на Kaggle: файл={out_path.name}, competition={competition!r}"
    )
    main_mod._do_kaggle_submit(
        str(д),
        competition,
        f"beam gap 4096 (limit={args.limit or 'all'})",
    )


if __name__ == "__main__":
    main()

