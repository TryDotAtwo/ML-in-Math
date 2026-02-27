# -*- coding: utf-8 -*-
"""Логирование результатов экспериментов в runs/experiment_results.jsonl для последующего анализа и выводов по гипотезам."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


RESULTS_FILE = Path("runs/experiment_results.jsonl")


def log_experiment(
    *,
    script: str = "main",
    command: str = "solve",
    method: str,
    test_path: str,
    out_path: str,
    score: int,
    n_rows: int,
    baseline_score: Optional[int] = None,
    **extra: Any,
) -> None:
    """Добавляет одну строку в runs/experiment_results.jsonl. Вызывать после solve или evaluate."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "script": script,
        "command": command,
        "method": method,
        "test_path": str(test_path),
        "out_path": str(out_path),
        "score": int(score),
        "n_rows": int(n_rows),
    }
    if baseline_score is not None:
        row["baseline_score"] = int(baseline_score)
        row["gain_vs_baseline"] = int(baseline_score - score)
    for k, v in extra.items():
        if v is not None and k not in row:
            row[k] = v
    try:
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def log_evaluate(
    *,
    script: str = "main",
    test_path: str,
    submission_path: str,
    baseline_total: int,
    submission_total: int,
    n_rows: int,
    **extra: Any,
) -> None:
    """Лог результата evaluate: baseline_total, submission_total, gain."""
    log_experiment(
        script=script,
        command="evaluate",
        method="evaluate",
        test_path=test_path,
        out_path=submission_path,
        score=submission_total,
        n_rows=n_rows,
        baseline_score=baseline_total,
        **extra,
    )


def analyze_results(max_entries: int = 50) -> str:
    """Читает последние записи из runs/experiment_results.jsonl, возвращает краткую сводку для вывода.
    Быстрый автоанализ без записи в md."""
    if not RESULTS_FILE.exists():
        return "Лог пуст: runs/experiment_results.jsonl не найден."
    lines = []
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
    except OSError:
        return "Не удалось прочитать лог."
    if not lines:
        return "Лог пуст."
    recent = lines[-max_entries:]
    rows = []
    for line in recent:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    # Приоритет: evaluate (есть baseline_score), иначе solve
    evals = [r for r in rows if r.get("command") == "evaluate" and "baseline_score" in r]
    solves = [r for r in rows if r.get("command") == "solve"]
    out = ["--- Анализ (последние записи) ---"]
    ref_baseline = 158680
    ref_notebook = 89980
    ref_beam = 91584
    if evals:
        last = evals[-1]
        sc = last.get("score", 0)
        base = last.get("baseline_score", 0)
        gain = last.get("gain_vs_baseline", 0)
        out.append(f"Последний evaluate: score={sc}  baseline={base}  gain={gain:+d}  (improved/worse в логе)")
        if gain < 0:
            out.append(f"  → метод хуже baseline на {-gain} ходов.")
        else:
            out.append(f"  → метод лучше baseline на {gain} ходов.")
        out.append(f"  Эталоны: notebook≈{ref_notebook}  beam≈{ref_beam}  baseline≈{ref_baseline}")
    if solves:
        last_solve = solves[-1]
        out.append(f"Последний solve: method={last_solve.get('method')}  score={last_solve.get('score')}  n_rows={last_solve.get('n_rows')}")
    if len(evals) > 1:
        out.append(f"Всего evaluate в логе: {len(evals)}; решено прогонов: {len(solves)}")
    return "\n".join(out)
