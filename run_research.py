#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Единый оркестратор исследований Pancake-проекта с поддержкой resume.

Запуск:
  python run_research.py
  python run_research.py --profile quick --limit 300
  python run_research.py --profile full --out-dir runs/research_full

Что делает:
  1) Строит submission для нескольких доступных методов (baseline, beam,
     notebook, unified, beam+LD, beam+singletons).
  2) Проверяет корректность каждого решения (check_steps).
  3) Сравнивает каждый метод с baseline.
  4) Собирает merged-best (лучшее решение по каждому id) и оценивает.
  5) Сохраняет state.json для resume: можно прервать Ctrl-C и запустить снова.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import psutil
import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from src.core import (
    apply_moves,
    is_solved,
    moves_len,
    moves_to_str,
    pancake_sort_moves,
    parse_permutation,
)
from src.crossings import solve_baseline_then_beam, solve_unified
from src.heuristics import (
    beam_improve_or_baseline_h,
    make_h,
    make_h_ld,
    make_h_singleton_tiebreak,
)
from src.notebook_search import get_solver
from src.submission import check_steps, evaluate_submission_vs_baseline

DEFAULT_TEST = "baseline/sample_submission.csv"

# Жёсткий лимит памяти: 20 ГБ
MEMORY_LIMIT_BYTES = 20 * 1024**3
# Как часто гарантированно логировать прогресс (даже если мало строк решено), сек
PROGRESS_TIME_LIMIT_SEC = 5.0

try:
    _PROC = psutil.Process(os.getpid())
except Exception:
    _PROC = None


def _get_rss_bytes() -> Optional[int]:
    """Текущее потребление ОЗУ процессом (rss) или None, если недоступно."""
    if _PROC is None:
        return None
    try:
        return _PROC.memory_info().rss
    except Exception:
        return None


# ─────────────────── helpers ───────────────────


@dataclass(frozen=True)
class MethodCfg:
    name: str
    kind: str
    params: dict = field(default_factory=dict)


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _read_test_df(test_path: Path) -> pd.DataFrame:
    df = pd.read_csv(test_path)
    if "id" not in df.columns or "permutation" not in df.columns:
        raise ValueError(f"{test_path} must contain columns: id, permutation")
    out = df[["id", "permutation"]].copy()
    out["id"] = out["id"].astype(int)
    out["n"] = out["permutation"].apply(lambda x: len(parse_permutation(x)))
    return out.sort_values("id").reset_index(drop=True)


# ─────────────────── method list ───────────────────


def _method_list(profile: str, include_rl: bool, models_dir: Path) -> List[MethodCfg]:
    quick = [
        MethodCfg("baseline", "baseline"),
        MethodCfg("beam_64x64", "beam", {"beam_width": 64, "depth": 64}),
        MethodCfg("beam_64x64_LD", "beam_custom_h", {"beam_width": 64, "depth": 64, "h_factory": "ld"}),
        MethodCfg("beam_64x64_singleton", "beam_custom_h", {"beam_width": 64, "depth": 64, "h_factory": "singleton"}),
        MethodCfg("notebook_v3_1_t3", "notebook", {"solver": "v3_1", "treshold": 3.0}),
        MethodCfg("notebook_v3_5_t3", "notebook", {"solver": "v3_5", "treshold": 3.0}),
        MethodCfg(
            "unified_nb_v3_1_beam_32x48",
            "unified",
            {"use_notebook_baseline": True, "notebook_solver": "v3_1", "treshold": 3.0,
             "beam_width": 32, "depth": 48},
        ),
    ]
    extra_full = [
        MethodCfg("beam_128x128", "beam", {"beam_width": 128, "depth": 128}),
        MethodCfg("beam_128x128_LD", "beam_custom_h", {"beam_width": 128, "depth": 128, "h_factory": "ld"}),
        MethodCfg("beam_128x128_singleton", "beam_custom_h", {"beam_width": 128, "depth": 128, "h_factory": "singleton"}),
        MethodCfg("notebook_v4_t2_6", "notebook", {"solver": "v4", "treshold": 2.6}),
    ]
    methods = list(quick) if profile == "quick" else quick + extra_full
    if include_rl and models_dir.exists():
        try:
            from src.ml import solve_with_rl_or_baseline  # noqa: F401
            methods.append(MethodCfg("rl_or_baseline", "rl", {"models_dir": str(models_dir)}))
        except Exception:
            _log("RL-модуль недоступен (torch?), пропускаю rl_or_baseline.")
    return methods


# ─────────────────── solve one perm ───────────────────


def _solve_notebook(perm: List[int], solver_name: str, treshold: float) -> List[int]:
    func, default_t = get_solver(solver_name)
    result = func(perm, treshold if treshold is not None else default_t)
    first = result[0]
    return list(first[0]) if isinstance(first, tuple) else list(first)


def _make_custom_h(name: str) -> Callable:
    if name == "ld":
        return make_h_ld()
    if name == "singleton":
        return make_h_singleton_tiebreak(make_h(0.0))
    return make_h(0.0)


def _solve_one(perm: List[int], cfg: MethodCfg, overrides: Optional[dict] = None) -> List[int]:
    """Решить одну perm указанным методом.
    overrides (при наличии) переопределяют параметры cfg.params во время выполнения
    (например, уменьшенный beam_width/depth при превышении лимита памяти).
    """
    p = dict(cfg.params)
    if overrides:
        p.update(overrides)
    if cfg.kind == "baseline":
        return pancake_sort_moves(perm)

    if cfg.kind == "beam":
        return solve_baseline_then_beam(
            perm,
            beam_width=int(p.get("beam_width", 64)),
            depth=int(p.get("depth", 64)),
            alpha=float(p.get("alpha", 0.0)),
            w=float(p.get("w", 0.5)),
        )

    if cfg.kind == "beam_custom_h":
        h_fn = _make_custom_h(str(p.get("h_factory", "gap")))
        return beam_improve_or_baseline_h(
            perm,
            baseline_moves_fn=pancake_sort_moves,
            h_fn=h_fn,
            beam_width=int(p.get("beam_width", 64)),
            depth=int(p.get("depth", 64)),
            w=float(p.get("w", 0.5)),
            prune_best_g_each_layer=True,
        )

    if cfg.kind == "notebook":
        return _solve_notebook(
            perm,
            solver_name=str(p.get("solver", "v3_1")),
            treshold=float(p.get("treshold", 3.0)),
        )

    if cfg.kind == "unified":
        return solve_unified(
            perm,
            use_notebook_baseline=bool(p.get("use_notebook_baseline", True)),
            notebook_solver=str(p.get("notebook_solver", "v3_1")),
            treshold=float(p.get("treshold", 3.0)),
            beam_width=int(p.get("beam_width", 32)),
            depth=int(p.get("depth", 48)),
            alpha=float(p.get("alpha", 0.0)),
            w=float(p.get("w", 0.5)),
        )

    if cfg.kind == "rl":
        from src.ml import solve_with_rl_or_baseline
        return solve_with_rl_or_baseline(perm, str(p.get("models_dir", "runs/rl_models")))

    raise ValueError(f"Unknown method kind: {cfg.kind}")


# ─────────────────── state persistence ───────────────────


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "methods": {}, "updated_at": _ts()}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict) -> None:
    state["updated_at"] = _ts()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ─────────────────── run / evaluate / merge ───────────────────


def _run_method(
    test_df: pd.DataFrame,
    cfg: MethodCfg,
    submission_path: Path,
    log_every: int,
) -> dict:
    existing_ids: set = set()
    wrote_header = False
    if submission_path.exists():
        try:
            prev = pd.read_csv(submission_path)
            if "id" in prev.columns:
                existing_ids = set(prev["id"].astype(int).tolist())
                wrote_header = True
        except Exception:
            pass

    total = len(test_df)
    _log(f"  [{cfg.name}] задач: {total}, уже решено: {len(existing_ids)}")
    batch: List[dict] = []
    t0 = time.time()
    last_log_ts = t0
    new_solved = 0
    max_rss = 0
    high_mem_events = 0
    row_times: List[float] = []

    # Для beam-подобных методов позволяем динамически уменьшать beam_width/depth
    runtime_overrides: dict = {}

    for i, row in enumerate(test_df.itertuples(index=False), start=1):
        rid = int(row.id)
        if rid in existing_ids:
            continue
        row_t0 = time.time()
        perm = parse_permutation(row.permutation)
        moves = _solve_one(perm, cfg, overrides=runtime_overrides or None)
        batch.append({"id": rid, "solution": moves_to_str(moves)})
        new_solved += 1
        row_times.append(time.time() - row_t0)

        # Обновляем максимум по памяти и при необходимости ужимаем beam
        rss = _get_rss_bytes()
        if rss is not None:
            if rss > max_rss:
                max_rss = rss
            if rss > MEMORY_LIMIT_BYTES and cfg.kind in ("beam", "beam_custom_h", "unified"):
                high_mem_events += 1
                old_bw = int(runtime_overrides.get("beam_width", cfg.params.get("beam_width", 64)))
                old_depth = int(runtime_overrides.get("depth", cfg.params.get("depth", 64)))
                new_bw = max(16, old_bw // 2)
                new_depth = max(24, old_depth // 2)
                runtime_overrides["beam_width"] = new_bw
                runtime_overrides["depth"] = new_depth
                _log(
                    f"  [{cfg.name}] HIGH MEM: {rss / 1024**3:.1f} GB > 20 GB → "
                    f"beam_width {old_bw}->{new_bw}, depth {old_depth}->{new_depth}"
                )

        if len(batch) >= 100:
            pd.DataFrame(batch).to_csv(
                submission_path, mode="a", header=not wrote_header, index=False,
            )
            wrote_header = True
            batch.clear()
            gc.collect()

        now = time.time()
        need_log_by_count = (i % max(1, log_every) == 0)
        need_log_by_time = (now - last_log_ts >= PROGRESS_TIME_LIMIT_SEC)
        if need_log_by_count or need_log_by_time:
            elapsed = now - t0
            speed = new_solved / max(elapsed, 0.001)
            mem_str = ""
            if rss is not None:
                mem_str = f" | mem={rss / 1024**3:.1f} GB max={max_rss / 1024**3:.1f} GB"
            _log(f"  [{cfg.name}] {i}/{total} | new={new_solved} | {speed:.1f} row/s{mem_str}")
            last_log_ts = now

    if batch:
        pd.DataFrame(batch).to_csv(
            submission_path, mode="a", header=not wrote_header, index=False,
        )

    final_df = pd.read_csv(submission_path)[["id", "solution"]].copy()
    final_df["id"] = final_df["id"].astype(int)
    final_df = final_df.drop_duplicates(subset=["id"], keep="last").sort_values("id")
    final_df.to_csv(submission_path, index=False)
    _log(f"  [{cfg.name}] готово: {len(final_df)} строк → {submission_path.name}")

    # Сводные метрики по времени и памяти
    rows_processed = len(row_times)
    mean_row_time = sum(row_times) / rows_processed if rows_processed else 0.0
    max_row_time = max(row_times) if rows_processed else 0.0
    return {
        "max_rss_bytes": int(max_rss),
        "max_rss_gb": (max_rss / 1024**3) if max_rss else 0.0,
        "high_memory_events": int(high_mem_events),
        "mean_row_time_sec": float(mean_row_time),
        "max_row_time_sec": float(max_row_time),
        "rows_processed": int(rows_processed),
    }


def _evaluate_method(test_df: pd.DataFrame, submission_path: Path) -> dict:
    sub_df = pd.read_csv(submission_path)
    merged = sub_df.merge(test_df[["id", "permutation"]], on="id", how="left")
    wrong_ids = check_steps(merged)
    stats = evaluate_submission_vs_baseline(
        test_df[["id", "permutation"]].copy(),
        sub_df[["id", "solution"]].copy(),
        baseline_moves_fn=pancake_sort_moves,
        log_every=0,
    )
    stats["wrong_solutions"] = len(wrong_ids)
    stats["score_from_submission"] = int(sub_df["solution"].map(moves_len).sum())
    return stats


def _build_merged_best(submission_files: Dict[str, Path], out_path: Path) -> pd.DataFrame:
    best: Dict[int, tuple] = {}
    for method, path in submission_files.items():
        df = pd.read_csv(path)[["id", "solution"]].copy()
        df["id"] = df["id"].astype(int)
        for row in df.itertuples(index=False):
            rid = int(row.id)
            sol = str(row.solution) if row.solution is not None else ""
            ln = moves_len(sol)
            cur = best.get(rid)
            if cur is None or ln < cur[1]:
                best[rid] = (sol, ln, method)
    out = pd.DataFrame(
        [{"id": rid, "solution": v[0], "source": v[2]} for rid, v in sorted(best.items())]
    )
    out.to_csv(out_path, index=False)
    return out


# ─────────────────── report ───────────────────


def _format_table(summary_df: pd.DataFrame) -> str:
    """Markdown-таблица без зависимости от tabulate."""
    if summary_df.empty:
        return "_No completed methods yet._"
    cols = list(summary_df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for row in summary_df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join([header, sep] + rows)


def _write_report(report_path: Path, summary_df: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Research Run Report",
        "",
        f"- timestamp: `{_ts()}`",
        f"- profile: `{args.profile}`",
        f"- test: `{args.test}`",
        f"- limit: `{args.limit}`",
        "",
        "## Ranking by score (lower is better)",
        "",
        _format_table(summary_df.sort_values("score")),
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _print_summary(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return
    _log("=" * 60)
    _log("ИТОГИ (score — сумма ходов, меньше = лучше):")
    _log("-" * 60)
    for row in summary_df.sort_values("score").itertuples(index=False):
        gain_str = f"{row.gain_vs_baseline:+d}" if row.gain_vs_baseline else "0"
        wrong_str = f" WRONG={row.wrong_solutions}" if row.wrong_solutions else ""
        _log(f"  {row.method:<35s} score={row.score:>8d}  gain={gain_str:>7s}{wrong_str}")
    _log("=" * 60)


# ─────────────────── main ───────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Пакетный прогон методов, сравнение с baseline, сборка merged-best. Resume через state.json."
    )
    parser.add_argument("--test", default=DEFAULT_TEST)
    parser.add_argument("--out-dir", default="runs/research")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--include-rl", action="store_true")
    parser.add_argument("--rl-models-dir", default="runs/rl_models")
    parser.add_argument("--reset", action="store_true", help="Сбросить state и начать с нуля")
    args = parser.parse_args()

    test_path = Path(args.test)
    if not test_path.exists():
        print(f"ОШИБКА: файл теста не найден: {test_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    submissions_dir = out_dir / "submissions"
    metrics_dir = out_dir / "metrics"
    reports_dir = out_dir / "reports"
    state_path = out_dir / "state.json"
    for d in [out_dir, submissions_dir, metrics_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.reset and state_path.exists():
        state_path.unlink()
        _log("state.json удалён (--reset)")
    state = _load_state(state_path)

    test_df = _read_test_df(test_path)
    if args.limit:
        test_df = test_df.head(args.limit).copy()

    methods = _method_list(args.profile, args.include_rl, Path(args.rl_models_dir))

    _log(f"Тест: {test_path} ({len(test_df)} строк) | профиль: {args.profile}")
    _log(f"Методов: {len(methods)} → " + ", ".join(m.name for m in methods))
    _log("")

    completed: Dict[str, Path] = {}

    for cfg in methods:
        rec = state["methods"].get(cfg.name, {})
        sub_path = submissions_dir / f"{cfg.name}.csv"
        met_path = metrics_dir / f"{cfg.name}.json"

        if rec.get("status") == "done" and sub_path.exists() and met_path.exists():
            _log(f"[{cfg.name}] ✓ уже выполнен (score={rec.get('score')}), пропускаю")
            completed[cfg.name] = sub_path
            continue

        state["methods"][cfg.name] = {
            "status": "running", "started_at": _ts(),
            "submission_path": str(sub_path), "metrics_path": str(met_path),
        }
        _save_state(state_path, state)

        try:
            _log(f"[{cfg.name}] ЗАПУСК...")
            run_info = _run_method(test_df, cfg, sub_path, log_every=args.log_every)

            _log(f"  [{cfg.name}] оценка vs baseline...")
            stats = _evaluate_method(test_df, sub_path)
            stats.update(run_info)
            met_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

            score = int(stats.get("score_from_submission", 0))
            gain = int(stats.get("total_gain", 0))
            wrong = int(stats.get("wrong_solutions", 0))
            max_rss_gb = float(stats.get("max_rss_gb", 0.0))
            high_mem_events = int(stats.get("high_memory_events", 0))

            state["methods"][cfg.name].update({
                "status": "done", "finished_at": _ts(),
                "score": score,
                "gain_vs_baseline": gain,
                "wrong_solutions": wrong,
                "max_rss_gb": max_rss_gb,
                "high_memory_events": high_mem_events,
            })
            _save_state(state_path, state)
            completed[cfg.name] = sub_path

            _log(f"  [{cfg.name}] ✓ score={score}  gain_vs_baseline={gain:+d}  wrong={wrong}")

        except Exception as exc:
            state["methods"][cfg.name].update({
                "status": "failed", "finished_at": _ts(),
                "error": f"{type(exc).__name__}: {exc}",
            })
            _save_state(state_path, state)
            _log(f"  [{cfg.name}] ✗ ОШИБКА: {type(exc).__name__}: {exc}")
            traceback.print_exc()

        _log("")
        gc.collect()

    # ── merged-best ──
    if completed:
        _log("Собираю merged_best.csv (лучшее решение по каждому id из всех методов)...")
        merged_path = submissions_dir / "merged_best.csv"
        _build_merged_best(completed, merged_path)
        merged_stats = _evaluate_method(test_df, merged_path)
        (metrics_dir / "merged_best.json").write_text(
            json.dumps(merged_stats, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        state["methods"]["merged_best"] = {
            "status": "done", "finished_at": _ts(),
            "submission_path": str(merged_path),
            "metrics_path": str(metrics_dir / "merged_best.json"),
            "score": int(merged_stats.get("score_from_submission", 0)),
            "gain_vs_baseline": int(merged_stats.get("total_gain", 0)),
            "wrong_solutions": int(merged_stats.get("wrong_solutions", 0)),
        }
        _save_state(state_path, state)
        _log(f"  merged_best: score={state['methods']['merged_best']['score']}")

    # ── summary ──
    summary_rows = []
    for name, rec in state.get("methods", {}).items():
        if rec.get("status") != "done":
            continue
        summary_rows.append({
            "method": name,
            "score": rec.get("score"),
            "gain_vs_baseline": rec.get("gain_vs_baseline"),
            "wrong_solutions": rec.get("wrong_solutions"),
        })
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("score")
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(
        summary_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8",
    )
    _write_report(reports_dir / "latest_report.md", summary_df, args)

    _log("")
    _print_summary(summary_df)
    _log(f"Артефакты: {out_dir}/")
    _log(f"Для resume: python run_research.py --profile {args.profile}" +
         (f" --limit {args.limit}" if args.limit else ""))


if __name__ == "__main__":
    main()
