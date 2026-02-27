#!/usr/bin/env python3
"""По submission CSV и (при необходимости) тесту строит сводку и таблицу по n: score, rows, solved, len_min/max/med/mean."""
import argparse
import sys
from pathlib import Path

# корень проекта
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.core.permutation import parse_permutation


def solution_length(s: str) -> int:
    """Число ходов в строке решения (R2.R5.R3 -> 3)."""
    if not isinstance(s, str) or not s.strip():
        return 0
    return s.count(".") + 1


def main():
    p = argparse.ArgumentParser(description="Сводка по submission: таблица по n (score, rows, solved, len_min/max/med/mean)")
    p.add_argument("--submission", "-s", default="submission.csv", help="CSV с колонками id, solution")
    p.add_argument(
        "--test",
        "-t",
        default="baseline/sample_submission.csv",
        help="CSV с id, permutation (если в submission нет permutation — для получения n)",
    )
    p.add_argument(
        "--n-list",
        nargs="*",
        type=int,
        default=[5, 12, 15, 16, 20, 25, 30, 35, 40, 45, 50, 75, 100],
        help="Значения n для строк таблицы",
    )
    args = p.parse_args()

    sub = pd.read_csv(args.submission)
    if "solution" not in sub.columns or "id" not in sub.columns:
        print("В submission нужны колонки id и solution.", file=sys.stderr)
        sys.exit(1)

    sub["len"] = sub["solution"].fillna("").apply(solution_length)

    if "permutation" in sub.columns:
        sub["n"] = sub["permutation"].apply(lambda x: len(parse_permutation(x)))
    else:
        test_path = Path(args.test)
        if not test_path.exists():
            print(f"Файл теста не найден: {test_path}. Добавьте --test или используйте submission с колонкой permutation.", file=sys.stderr)
            sys.exit(1)
        test_df = pd.read_csv(test_path)[["id", "permutation"]]
        sub = sub.merge(test_df, on="id", how="left")
        sub["n"] = sub["permutation"].apply(lambda x: len(parse_permutation(x)))

    total_rows = len(sub)
    total_score = sub["len"].sum()
    solved = (sub["len"] > 0).sum()

    print(f"Rows: {total_rows} | Solved: {solved} | Score: {int(total_score)}")
    print()

    n_list = sorted(args.n_list)
    agg = (
        sub.groupby("n", as_index=False)
        .agg(
            score=("len", "sum"),
            rows=("id", "count"),
            solved=("len", lambda x: (x > 0).sum()),
            len_min=("len", "min"),
            len_max=("len", "max"),
            len_med=("len", "median"),
            len_mean=("len", "mean"),
        )
        .astype({"score": int, "rows": int, "solved": int})
    )
    agg = agg[agg["n"].isin(n_list)].sort_values("n")
    agg["len_med"] = agg["len_med"].round(1)
    agg["len_mean"] = agg["len_mean"].round(1)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
