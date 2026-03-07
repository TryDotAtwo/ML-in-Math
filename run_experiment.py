#!/usr/bin/env python3
"""
Единая точка входа для экспериментов:

- research: оркестратор методов (baseline / beam / notebook / unified / RL) с resume.
- rl: RL-пайплайн (train / solve / evaluate / submit / full).
- stats: сводка по submission (таблица по n).

Примеры:
  python run_experiment.py research --profile quick --out-dir runs/research
  python run_experiment.py rl full --train --test baseline/sample_submission.csv --models runs/rl_models --out submission.csv --evaluate --submit
  python run_experiment.py stats --submission submission.csv
"""

from __future__ import annotations

import sys


def _dispatch() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(__doc__ or "", flush=True)
        print("Использование:", flush=True)
        print("  python run_experiment.py research [аргументы run_research]", flush=True)
        print("  python run_experiment.py rl [аргументы run_rl]", flush=True)
        print("  python run_experiment.py stats [аргументы submission_stats]", flush=True)
        sys.exit(0)

    mode = sys.argv[1]
    rest = sys.argv[2:]

    if mode == "research":
        from scripts.runners import run_research as _mod

        sys.argv = ["run_research.py", *rest]
        _mod.main()
        return

    if mode == "rl":
        from scripts.runners import run_rl as _mod

        sys.argv = ["run_rl.py", *rest]
        _mod.main()
        return

    if mode == "stats":
        from scripts.runners import submission_stats as _mod

        sys.argv = ["submission_stats.py", *rest]
        _mod.main()
        return

    print(f"Неизвестный режим: {mode!r}. Ожидается один из: research, rl, stats.", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    _dispatch()

