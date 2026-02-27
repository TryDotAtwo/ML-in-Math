#!/usr/bin/env python3
"""Запуск solve в режимах, повторяющих оригиналы (без beam в блокноте, без блокнота в 91584)."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    p = argparse.ArgumentParser(
        description="Solve: notebook (v4 only, target 89980) или beam (baseline+beam, target 91584)"
    )
    p.add_argument("--limit", type=int, default=None, help="Макс. число строк (для теста)")
    p.add_argument("--out", default="submission.csv", help="Выходной CSV")
    p.add_argument(
        "--mode",
        choices=["notebook", "beam", "crossing-memory", "crossing-quality"],
        default="notebook",
        help=(
            "notebook: только v4, treshold=2.6 (как в блокноте, цель 89980). "
            "beam: только baseline+beam 128x128 (как в 91584, цель 91584). "
            "crossing-*: экспериментально блокнот+beam (больше памяти)."
        ),
    )
    args = p.parse_args()

    import main as main_mod

    if args.mode == "notebook":
        # Блокнот: только pancake_sort_v4, treshold=2.6, без beam (см. копия_блокнота, process_row(perm, pancake_sort_v4, treshold=2.6))
        ns = argparse.Namespace(
            test="baseline/sample_submission.csv",
            out=args.out,
            method="notebook",
            solver="v4",
            treshold=2.6,
            notebook_baseline=False,
            beam_width=128,
            depth=128,
            alpha=0.0,
            w=0.5,
            limit=args.limit,
            max_n=None,
            notebook_max_states=None,
            notebook_baseline_after_id=2200,
            submit=True,
            competition=os.environ.get("KAGGLE_COMPETITION", main_mod._DEFAULT_KAGGLE_COMPETITION),
            message="",
        )
        mode_desc = "notebook: только v4, treshold=2.6; id>=2300 — baseline (обход MemoryError)"
    elif args.mode == "beam":
        # pancake_91584: только baseline + beam с gap-эвристикой, без блокнота (submission_gap.csv → 91584)
        ns = argparse.Namespace(
            test="baseline/sample_submission.csv",
            out=args.out,
            method="beam",
            solver="v3_1",
            treshold=3,
            notebook_baseline=False,
            beam_width=128,
            depth=128,
            alpha=0.0,
            w=0.5,
            limit=args.limit,
            max_n=None,
            notebook_max_states=None,
            submit=True,
            competition=os.environ.get("KAGGLE_COMPETITION", main_mod._DEFAULT_KAGGLE_COMPETITION),
            message="",
        )
        mode_desc = "beam: baseline + beam 128x128, gap (оригинал 91584, цель 91584)"
    elif args.mode == "crossing-memory":
        ns = argparse.Namespace(
            test="baseline/sample_submission.csv",
            out=args.out,
            method="crossing-notebook-beam",
            solver="v3_1",
            treshold=3,
            notebook_baseline=False,
            beam_width=32,
            depth=48,
            alpha=0.0,
            w=0.5,
            limit=args.limit,
            max_n=40,
            notebook_max_states=500_000,
            submit=True,
            competition=os.environ.get("KAGGLE_COMPETITION", main_mod._DEFAULT_KAGGLE_COMPETITION),
            message="",
        )
        mode_desc = "crossing (memory): v3_1+beam 32x48, max_n=40 (~8–12 GB)"
    else:
        ns = argparse.Namespace(
            test="baseline/sample_submission.csv",
            out=args.out,
            method="crossing-notebook-beam",
            solver="v4",
            treshold=10,
            notebook_baseline=False,
            beam_width=96,
            depth=96,
            alpha=0.0,
            w=0.5,
            limit=args.limit,
            max_n=80,
            notebook_max_states=2_000_000,
            submit=True,
            competition=os.environ.get("KAGGLE_COMPETITION", main_mod._DEFAULT_KAGGLE_COMPETITION),
            message="",
        )
        mode_desc = "crossing (quality): v4+beam 96x96, max_n=80 (~16–20 GB)"

    print("Цели: блокнот 89980 (только v4), 91584 — baseline+beam (без блокнота).")
    print(f"Режим: {mode_desc}\n")
    main_mod.cmd_solve(ns)

if __name__ == "__main__":
    main()
