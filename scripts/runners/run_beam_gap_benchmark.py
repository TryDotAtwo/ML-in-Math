#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Быстрое тестирование beam_gap_4096: по одной перестановке на каждое n,
замер памяти и подбор батча для максимальной скорости.

Использование:
  python -m scripts.runners.run_beam_gap_benchmark
  python -m scripts.runners.run_beam_gap_benchmark --test baseline/sample_submission.csv --quick
  python -m scripts.runners.run_beam_gap_benchmark --out-dir runs/beam_bench

Фазы:
  1) Решаем по 1 перестановке для каждого n от 1 до 100, замеряем время и пик памяти GPU.
  2) Для каждого n пробуем батчи 1, 2, 3, ..., 32 без пропусков; при OOM фиксируем предел.
  3) Для каждого n выбираем батч с максимальной скоростью (row/s), не превышающий лимит памяти.
  Для n, отсутствующих в тесте, перестановки генерируются случайно.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import parse_permutation, pancake_sort_moves  # noqa: E402
from src.heuristics.gpu_beam import (
    beam_improve_or_baseline_h_gpu,
    beam_improve_batch_h_gpu,
)  # noqa: E402

BEAM_WIDTH = 4096
DEPTH = 4096
W = 0.5
# Батчи от 1 до 32 без пропусков
BATCH_SIZES_TO_TRY = list(range(1, 33))
# Лимит памяти GPU в GiB — батчи с пиком выше не считаем допустимыми
GPU_LIMIT_GIB = 7.5
N_MIN, N_MAX = 1, 100


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _get_gpu_memory_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def _reset_peak_memory() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except Exception:
        pass


def _synchronize() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def run_single(perm: List[int], device: str | None) -> float:
    """Один запуск single beam, возвращает время в секундах."""
    _reset_peak_memory()
    _synchronize()
    t0 = time.perf_counter()
    beam_improve_or_baseline_h_gpu(
        perm,
        baseline_moves_fn=pancake_sort_moves,
        beam_width=BEAM_WIDTH,
        depth=DEPTH,
        w=W,
        device=device,
        log=False,
    )
    _synchronize()
    return time.perf_counter() - t0


def run_batch(perms: List[List[int]], device: str | None) -> float:
    """Один запуск batch beam, возвращает время в секундах."""
    _reset_peak_memory()
    _synchronize()
    t0 = time.perf_counter()
    beam_improve_batch_h_gpu(
        perms,
        baseline_moves_fn=pancake_sort_moves,
        beam_width=BEAM_WIDTH,
        depth=DEPTH,
        w=W,
        device=device,
        log=False,
    )
    _synchronize()
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк beam_gap: память и оптимальный батч по n.")
    parser.add_argument(
        "--test",
        default="baseline/sample_submission.csv",
        help="CSV с id, permutation.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda / cpu (по умолчанию cuda при наличии).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Только подмножество n: 5, 12, 25, 50, 75, 100 (если есть в тесте).",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/beam_bench",
        help="Каталог для JSON/CSV результатов.",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=None,
        help="Максимальное n для теста (по умолчанию все из теста).",
    )
    args = parser.parse_args()

    test_path = Path(args.test)
    if not test_path.exists():
        print(f"ОШИБКА: не найден {test_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(test_path)
    if "id" not in df.columns or "permutation" not in df.columns:
        print("ОШИБКА: нужны колонки id, permutation", file=sys.stderr)
        sys.exit(1)

    perms = [parse_permutation(p) for p in df["permutation"].to_list()]
    ns = [len(p) for p in perms]

    n_to_indices: Dict[int, List[int]] = {}
    for idx, n_val in enumerate(ns):
        n_to_indices.setdefault(n_val, []).append(idx)

    # n от 1 до 100 без пропусков; для отсутствующих в тесте генерируем перестановки
    n_values = list(range(N_MIN, N_MAX + 1))
    if args.max_n is not None:
        n_values = [n for n in n_values if n <= args.max_n]
    if args.quick:
        quick_set = {5, 12, 25, 50, 75, 100}
        n_values = [n for n in n_values if n in quick_set]

    max_batch = max(BATCH_SIZES_TO_TRY)
    rng = random.Random(42)

    def get_perms_for_n(n_val: int) -> List[List[int]]:
        if n_val in n_to_indices:
            idx_list = n_to_indices[n_val]
            return [list(perms[k]) for k in idx_list[:max_batch]]
        return [rng.sample(range(n_val), n_val) for _ in range(max_batch)]

    perms_by_n: Dict[int, List[List[int]]] = {n: get_perms_for_n(n) for n in n_values}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Тест: {test_path.name}, n: {N_MIN}..{N_MAX} ({len(n_values)} значений), батчи 1..{max_batch}")

    # --- Фаза 1: по одной перестановке на каждое n, время + память ---
    _log("Фаза 1: single beam по 1 перестановке на каждое n (время + память)")
    single_results: Dict[int, Dict[str, Any]] = {}
    for n_val in n_values:
        perm = perms_by_n[n_val][0]
        _reset_peak_memory()
        t = run_single(perm, args.device)
        mem_mb = _get_gpu_memory_mb()
        single_results[n_val] = {"time_sec": t, "memory_mb": mem_mb}
        _log(f"  n={n_val}  time={t:.3f}s  mem={mem_mb:.0f} MB")

    # --- Фаза 2: для каждого n пробуем батчи 1, 2, ..., 32 ---
    _log("Фаза 2: подбор батча по скорости и памяти (батчи 1..32)")
    batch_results: Dict[int, Dict[int, Dict[str, Any]]] = {n: {} for n in n_values}
    for n_val in n_values:
        perms_n = perms_by_n[n_val]
        for batch_sz in BATCH_SIZES_TO_TRY:
            if len(perms_n) < batch_sz:
                continue
            batch_perms = perms_n[: batch_sz]
            try:
                _reset_peak_memory()
                t = run_batch(batch_perms, args.device)
                mem_mb = _get_gpu_memory_mb()
                batch_results[n_val][batch_sz] = {
                    "time_sec": t,
                    "memory_mb": mem_mb,
                    "throughput": batch_sz / t if t > 0 else 0,
                    "oom": False,
                }
                _log(f"  n={n_val} batch={batch_sz}  time={t:.3f}s  mem={mem_mb:.0f} MB  {batch_sz/t:.1f} row/s")
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    batch_results[n_val][batch_sz] = {
                        "time_sec": None,
                        "memory_mb": None,
                        "throughput": None,
                        "oom": True,
                    }
                    _log(f"  n={n_val} batch={batch_sz}  OOM")
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                else:
                    raise

    # --- Рекомендованный батч по n: макс. throughput при памяти < GPU_LIMIT_GIB ---
    limit_mb = GPU_LIMIT_GIB * 1024
    recommended: Dict[int, int] = {}
    for n_val in n_values:
        best_batch = 1
        best_throughput = 0.0
        for batch_sz in BATCH_SIZES_TO_TRY:
            if batch_sz not in batch_results[n_val]:
                continue
            r = batch_results[n_val][batch_sz]
            if r.get("oom"):
                continue
            mem = r.get("memory_mb") or 0
            if mem > limit_mb:
                continue
            tp = r.get("throughput") or 0
            if tp > best_throughput:
                best_throughput = tp
                best_batch = batch_sz
        recommended[n_val] = best_batch

    # --- Вывод таблицы и сохранение ---
    _log("")
    _log("Рекомендованный батч по n (макс. скорость при памяти < {:.1f} GiB):".format(GPU_LIMIT_GIB))
    rows = []
    for n_val in n_values:
        b = recommended[n_val]
        r = batch_results[n_val].get(b, {})
        tp = r.get("throughput") or 0
        mem = r.get("memory_mb") or single_results.get(n_val, {}).get("memory_mb") or 0
        rows.append({"n": n_val, "batch": b, "throughput_row_s": round(tp, 2), "memory_mb": round(mem, 0)})
        _log(f"  n={n_val}  batch={b}  {tp:.1f} row/s  {mem:.0f} MB")

    out_csv = out_dir / "batch_recommendations.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    _log(f"Сохранено: {out_csv}")

    payload = {
        "single_results": single_results,
        "batch_results": {
            str(n): {str(bs): data for bs, data in data_n.items()}
            for n, data_n in batch_results.items()
        },
        "recommended": {str(k): v for k, v in recommended.items()},
        "gpu_limit_gib": GPU_LIMIT_GIB,
    }
    out_json = out_dir / "benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    _log(f"Сохранено: {out_json}")

    # Сводка порогов для batch_size_for_n (лестница по n)
    ladder_lines = _build_ladder_snippet(recommended, n_values)
    ladder_path = out_dir / "batch_ladder_snippet.py"
    with open(ladder_path, "w", encoding="utf-8") as f:
        f.write("# Вставь в run_beam_gap_4096.py вместо текущей лестницы\n\n")
        f.write("\n".join(ladder_lines))
    _log(f"Сохранено: {ladder_path}")
    _log("Готово.")


def _build_ladder_snippet(recommended: Dict[int, int], n_values: List[int]) -> List[str]:
    """Строит if/elif лестницу по рекомендованным батчам (склеивая подряд одинаковые)."""
    if not n_values:
        return ["def batch_size_for_n(n_val: int) -> int:", "    return 1"]
    n_sorted = sorted(n_values, reverse=True)  # 100, 75, 50, ...
    lines = ["def batch_size_for_n(n_val: int) -> int:"]
    prev_batch = None
    for n_val in n_sorted:
        b = recommended.get(n_val, 1)
        if b == prev_batch:
            continue
        prev_batch = b
        lines.append(f"    if n_val >= {n_val}:")
        lines.append(f"        return {b}")
    default_batch = recommended.get(min(n_values), 1)
    lines.append(f"    return {default_batch}")
    return lines


if __name__ == "__main__":
    main()
