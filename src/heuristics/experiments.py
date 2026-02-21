# -*- coding: utf-8 -*-
"""Эксперименты: выбор кейсов, грид по параметрам beam, полный прогон конфигураций."""

from __future__ import annotations

import os
import time
import random
import itertools
from typing import List, Callable, Iterable

import pandas as pd

from ..core.permutation import parse_permutation
from ..core.moves import apply_moves, moves_to_str
from ..core.baseline import pancake_sort_moves
from .h_functions import make_h
from .beam import beam_improve_or_baseline_h


def select_cases_per_n(
    df: pd.DataFrame,
    n_list: List[int],
    k: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """По каждому n из n_list выбирает k случайных строк из df (должна быть колонка n)."""
    rng = random.Random(seed)
    rows = []
    for n in n_list:
        sub = df[df["n"] == n]
        if len(sub) < k:
            raise ValueError(f"Not enough samples for n={n}: have {len(sub)}, need {k}")
        idxs = list(sub.index)
        rng.shuffle(idxs)
        rows.append(df.loc[idxs[:k]])
    return pd.concat(rows, axis=0).reset_index(drop=True)


def _log_print(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, flush=True)


def run_grid(
    mini_df: pd.DataFrame,
    *,
    alphas: List[float],
    ws: List[float],
    beam_widths: List[int],
    depths: List[int],
    baseline_moves_fn: Callable[[Iterable[int]], List[int]] = pancake_sort_moves,
    log: bool = True,
    log_each: int = 1,
    beam_log: bool = False,
    beam_log_every_layer: int = 5,
) -> pd.DataFrame:
    """Перебор комбинаций (alpha, w, beam_width, depth); для каждой — beam_improve на каждом кейсе."""
    rows: List[dict] = []
    total_cfg = len(alphas) * len(ws) * len(beam_widths) * len(depths)
    total_cases = len(mini_df)
    total_runs = total_cfg * total_cases

    _log_print(log, f"[grid] cases={total_cases} cfg_per_case={total_cfg} total_runs={total_runs}")

    parsed: List[tuple] = []
    for i in range(total_cases):
        row = mini_df.iloc[i]
        perm = parse_permutation(row.permutation)
        n = len(perm)
        base_len = len(baseline_moves_fn(perm))
        parsed.append((int(row.id), n, perm, base_len))

    run_idx = 0
    for case_i, (rid, n, perm, base_len) in enumerate(parsed):
        _log_print(log, f"\n[case {case_i+1}/{total_cases}] id={rid} n={n} base_len={base_len}")

        for cfg_i, (alpha, w, bw, d) in enumerate(
            itertools.product(alphas, ws, beam_widths, depths), start=1
        ):
            run_idx += 1
            do_cfg_log = log and (cfg_i % max(1, log_each) == 0)

            t0 = time.time()
            h_fn = make_h(alpha)
            sol = beam_improve_or_baseline_h(
                perm,
                baseline_moves_fn=baseline_moves_fn,
                h_fn=h_fn,
                beam_width=bw,
                depth=d,
                w=w,
                log=(beam_log and do_cfg_log),
                log_every_layer=beam_log_every_layer,
            )
            dt = time.time() - t0

            ok = apply_moves(perm, sol) == list(range(n))
            steps = len(sol)
            gain = base_len - steps

            if do_cfg_log:
                _log_print(
                    True,
                    f"[run {run_idx}/{total_runs}] cfg={cfg_i:03d}/{total_cfg} "
                    f"alpha={alpha} w={w} bw={bw} depth={d} -> steps={steps} gain={gain} ok={ok} t={dt:.3f}s",
                )

            rows.append({
                "id": rid,
                "n": n,
                "base_len": base_len,
                "alpha": alpha,
                "w": w,
                "beam_width": bw,
                "depth": d,
                "ok": ok,
                "steps": steps,
                "gain": gain,
                "time_sec": dt,
            })

    return pd.DataFrame(rows)


def full_eval_top_cfgs(
    test_df: pd.DataFrame,
    n_list: List[int],
    top_cfgs: List[dict],
    *,
    out_csv_path: str,
    baseline_moves_fn: Callable[[Iterable[int]], List[int]] = pancake_sort_moves,
    log: bool = True,
    log_every: int = 50,
) -> pd.DataFrame:
    """Полный прогон top_cfgs по всем кейсам с n в n_list. Сохранение в CSV с возможностью resume."""
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)

    if "n" not in test_df.columns:
        test_df = test_df.copy()
        test_df["n"] = test_df["permutation"].apply(lambda x: len(parse_permutation(x)))

    sub = test_df[test_df["n"].isin(n_list)].reset_index(drop=True)

    done = set()
    wrote_header = os.path.exists(out_csv_path)
    if wrote_header:
        try:
            prev = pd.read_csv(out_csv_path, usecols=["id", "cfg_idx"])
            done = set(
                zip(prev["id"].astype(int).tolist(), prev["cfg_idx"].astype(int).tolist())
            )
            if log:
                print(f"[resume] found existing file with {len(done)} completed runs", flush=True)
        except Exception as e:
            if log:
                print(f"[resume] could not read existing file safely: {e!r} (will append anyway)", flush=True)

    rows_cache = []
    t_global0 = time.time()
    total_cases = len(sub)
    total_runs = total_cases * len(top_cfgs)
    run_idx = 0
    skipped = 0

    for i in range(total_cases):
        rid = int(sub.loc[i, "id"])
        perm = parse_permutation(sub.loc[i, "permutation"])
        n = len(perm)

        base_moves = baseline_moves_fn(perm)
        base_len = len(base_moves)

        for cfg_j, cfg in enumerate(top_cfgs):
            run_idx += 1

            if (rid, cfg_j) in done:
                skipped += 1
                if log and (run_idx % log_every == 0):
                    elapsed = time.time() - t_global0
                    speed = (run_idx - skipped) / max(1e-9, elapsed)
                    _log_print(True, f"[full] {run_idx}/{total_runs} runs | skipped={skipped} | speed={speed:.3f} new_runs/s")
                continue

            alpha = float(cfg["alpha"])
            w = float(cfg["w"])
            bw = int(cfg["beam_width"])
            depth = int(cfg["depth"])

            t0 = time.time()
            status = "ok"
            err_txt = ""

            try:
                h_fn = make_h(alpha)
                moves = beam_improve_or_baseline_h(
                    perm,
                    baseline_moves_fn=baseline_moves_fn,
                    h_fn=h_fn,
                    beam_width=bw,
                    depth=depth,
                    w=w,
                    log=False,
                )
                if apply_moves(perm, moves) != list(range(n)):
                    moves = base_moves
                    status = "fallback_baseline"
            except Exception as e:
                moves = base_moves
                status = "error_fallback_baseline"
                err_txt = repr(e)

            dt = time.time() - t0
            steps = len(moves)
            gain = base_len - steps
            sol_str = moves_to_str(moves)

            row = {
                "id": rid,
                "n": n,
                "cfg_idx": cfg_j,
                "alpha": alpha,
                "w": w,
                "beam_width": bw,
                "depth": depth,
                "base_len": base_len,
                "ok": (status == "ok"),
                "steps": steps,
                "gain": gain,
                "time_sec": dt,
                "solution": sol_str,
                "status": status,
                "error": err_txt,
            }
            rows_cache.append(row)
            done.add((rid, cfg_j))

            if len(rows_cache) >= 200:
                pd.DataFrame(rows_cache).to_csv(
                    out_csv_path, mode="a", header=not wrote_header, index=False
                )
                wrote_header = True
                rows_cache.clear()

            if log and (run_idx % log_every == 0):
                elapsed = time.time() - t_global0
                new_done = run_idx - skipped
                speed = new_done / max(1e-9, elapsed)
                _log_print(
                    True,
                    f"[full] {run_idx}/{total_runs} runs | new={new_done} skipped={skipped} | "
                    f"n={n} cfg={cfg_j} steps={steps} gain={gain} status={status} | {speed:.3f} new_runs/s",
                )

    if rows_cache:
        pd.DataFrame(rows_cache).to_csv(
            out_csv_path, mode="a", header=not wrote_header, index=False
        )

    return pd.read_csv(out_csv_path)
