#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H_heuristic / H_grid: тестовый запуск beam search с шириной 2^12 (4096)
и "чистой" gap-эвристикой (alpha=0, без breakpoints).

Запускать из корня проекта:

  python -m scripts.hypotheses.H_heuristic_beam_gap_4096 --limit 100

По сути это обёртка над main.py solve --method beam с жёстко заданными
параметрами beam_width=4096, depth=4096, alpha=0.0, w=0.5.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# Корень проекта (ML in Math)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Тестовый beam-search: baseline + beam с gap-эвристикой, "
            "ширина луча 2^12=4096, глубина 4096."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Максимальное число строк из baseline/sample_submission.csv (по умолчанию 100).",
    )
    parser.add_argument(
        "--out",
        default="submission_beam_gap_4096.csv",
        help="Выходной CSV (id, solution).",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="После решения отправить сабмит на Kaggle (см. README, требует kaggle.json).",
    )
    args = parser.parse_args()

    import main as main_mod

    ns = argparse.Namespace(
        # Источник перестановок — тот же, что и во всех пайплайнах по умолчанию
        test="baseline/sample_submission.csv",
        out=args.out,
        method="beam",
        # Аргументы блокнота / crossing здесь не используются, но заполняем для совместимости
        solver="v3_1",
        treshold=3,
        notebook_baseline=False,
        notebook_max_states=None,
        notebook_baseline_after_id=None,
        # Параметры beam search: ширина и глубина 2^12, чистая gap-эвристика
        beam_width=4096,
        depth=4096,
        alpha=0.0,  # alpha=0 → make_h(alpha) = gap_h
        w=0.5,  # как в оригинальном 91584-пайплайне
        # Прочие служебные параметры main.cmd_solve
        limit=args.limit,
        max_n=None,
        rl_models="runs/rl_models",
        rl_device=None,
        submit=args.submit,
        competition=os.environ.get("KAGGLE_COMPETITION", main_mod._DEFAULT_KAGGLE_COMPETITION),
        message="beam 2^12 gap (test)",
    )

    main_mod.cmd_solve(ns)


if __name__ == "__main__":
    main()

