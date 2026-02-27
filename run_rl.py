#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт RL: обучение политики (BC + опционально PG), решение теста, оценка и сабмит.
Цель — минимальный суммарный скор (сумма длин решений).

Использование:
  python run_rl.py train --test baseline/sample_submission.csv --out-dir runs/rl_models
  python run_rl.py solve --test baseline/sample_submission.csv --models runs/rl_models --out submission.csv
  python run_rl.py evaluate --test baseline/sample_submission.csv --submission submission.csv
  python run_rl.py submit --file submission.csv
  python run_rl.py analyze   # сводка по runs/experiment_results.jsonl
  python run_rl.py full --train --test baseline/sample_submission.csv --models runs/rl_models --out submission.csv --evaluate --submit
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.core import parse_permutation, moves_to_str, pancake_sort_moves
from src.ml import (
    train_bc,
    train_bc_all_n,
    train_pg_epochs,
    load_policy_for_n,
    solve_with_rl_or_baseline,
)
from src.submission import evaluate_submission_vs_baseline, log_experiment, log_evaluate, analyze_results

_DEFAULT_TEST_PATH = "baseline/sample_submission.csv"
_DEFAULT_MODELS_DIR = "runs/rl_models"
_DEFAULT_KAGGLE_COMPETITION = "CayleyPy-pancake"
N_LIST = [5, 12, 15, 16, 20, 25, 30, 35, 40, 45, 50, 75, 100]


def _ensure_n(df: pd.DataFrame) -> pd.DataFrame:
    if "n" not in df.columns and "permutation" in df.columns:
        df = df.copy()
        df["n"] = df["permutation"].apply(lambda x: len(parse_permutation(x)))
    return df


def cmd_train(args: argparse.Namespace) -> None:
    """Обучить политики для каждого n: BC, опционально PG."""
    test_path = Path(args.test)
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        test_df = _ensure_n(test_df)
        n_list = sorted(test_df["n"].unique().tolist())
    else:
        n_list = args.n_list or N_LIST
    n_list = [int(x) for x in n_list]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training policies for n in {n_list}, save to {out_dir}")

    for n in n_list:
        save_path = out_dir / f"policy_n_{n}.pt"
        train_bc(
            n,
            num_trajectories=args.trajectories,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            save_path=save_path,
        )
        if args.pg_epochs and args.pg_epochs > 0:
            import torch
            from src.ml.policy import PolicyNet
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(save_path, map_location=device, weights_only=True)
            model = PolicyNet(
                n,
                emb_dim=ckpt.get("emb_dim", 32),
                hidden=ckpt.get("hidden", 128),
                num_layers=ckpt.get("num_layers", 2),
            )
            model.load_state_dict(ckpt["state_dict"])
            model = train_pg_epochs(
                model,
                n,
                num_rollouts=args.pg_rollouts,
                epochs=args.pg_epochs,
                lr=args.pg_lr,
                device=device,
                seed=args.seed,
            )
            torch.save(
                {
                    "n": n,
                    "emb_dim": ckpt.get("emb_dim", 32),
                    "hidden": ckpt.get("hidden", 128),
                    "num_layers": ckpt.get("num_layers", 2),
                    "state_dict": model.state_dict(),
                },
                save_path,
            )
    print("Done. Models saved to", out_dir)


def cmd_solve(args: argparse.Namespace) -> None:
    """Решить тест: RL-политика где есть модель, иначе baseline. Записать submission CSV."""
    import torch
    test_path = Path(args.test)
    if not test_path.exists():
        test_path = Path(_DEFAULT_TEST_PATH)
    if not test_path.exists():
        raise SystemExit(f"Test file not found: {args.test} and {_DEFAULT_TEST_PATH}")
    test_df = pd.read_csv(test_path)
    test_df = _ensure_n(test_df)
    models_dir = Path(args.models)
    device = torch.device(args.device) if args.device else None

    rows = []
    for i, row in test_df.iterrows():
        rid = int(row["id"])
        perm = parse_permutation(row["permutation"])
        if args.limit and len(rows) >= args.limit:
            break
        moves = solve_with_rl_or_baseline(perm, models_dir, device=device)
        rows.append({"id": rid, "solution": moves_to_str(moves)})
        if (len(rows)) % 200 == 0 and len(rows) > 0:
            print(f"  solved {len(rows)} ...", flush=True)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)
    total = out_df["solution"].fillna("").apply(
        lambda s: s.count(".") + 1 if isinstance(s, str) and s.strip() else 0
    ).sum()
    print(f"Saved {len(out_df)} rows to {out_path} | total moves (score): {int(total)}")
    log_experiment(
        script="run_rl",
        command="solve",
        method="rl",
        test_path=str(test_path),
        out_path=str(out_path),
        score=int(total),
        n_rows=len(out_df),
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Сравнить submission с baseline по тесту."""
    test_df = pd.read_csv(args.test)
    sub_df = pd.read_csv(args.submission)
    test_df = _ensure_n(test_df)
    stats = evaluate_submission_vs_baseline(
        test_df,
        sub_df,
        baseline_moves_fn=pancake_sort_moves,
        log_every=0,
    )
    print("--- Evaluation vs baseline ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    log_evaluate(
        script="run_rl",
        test_path=args.test,
        submission_path=args.submission,
        baseline_total=stats.get("baseline_total", 0),
        submission_total=stats.get("submission_total", 0),
        n_rows=len(sub_df),
        total_gain=stats.get("total_gain"),
        improved_cases=stats.get("improved_cases"),
        worse_cases=stats.get("worse_cases"),
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    """Быстро вывести сводку по runs/experiment_results.jsonl (последние записи, score vs baseline)."""
    print(analyze_results(max_entries=getattr(args, "max_entries", 50)), flush=True)


def cmd_submit(args: argparse.Namespace) -> None:
    """Отправить submission на Kaggle."""
    import main as main_mod
    main_mod._do_kaggle_submit(
        args.file,
        args.competition or os.environ.get("KAGGLE_COMPETITION", _DEFAULT_KAGGLE_COMPETITION),
        args.message or f"RL submission {Path(args.file).name}",
    )


def cmd_full(args: argparse.Namespace) -> None:
    """(Optional) train -> solve -> (optional) evaluate -> (optional) submit."""
    if getattr(args, "train", False):
        ns_train = argparse.Namespace(
            test=args.test,
            out_dir=args.models,
            n_list=None,
            trajectories=getattr(args, "trajectories", 5000),
            batch_size=getattr(args, "batch_size", 256),
            epochs=getattr(args, "epochs", 30),
            lr=getattr(args, "lr", 1e-3),
            seed=getattr(args, "seed", 42),
            pg_epochs=getattr(args, "pg_epochs", 0),
            pg_rollouts=getattr(args, "pg_rollouts", 500),
            pg_lr=getattr(args, "pg_lr", 1e-4),
        )
        cmd_train(ns_train)
    ns = argparse.Namespace(
        test=args.test,
        models=args.models,
        out=args.out,
        limit=args.limit,
        device=args.device,
    )
    cmd_solve(ns)
    if args.evaluate:
        ns_sub = argparse.Namespace(
            test=args.test,
            submission=args.out,
        )
        cmd_evaluate(ns_sub)
        print(analyze_results(), flush=True)
    if args.submit:
        ns_submit = argparse.Namespace(
            file=args.out,
            competition=args.competition or os.environ.get("KAGGLE_COMPETITION", _DEFAULT_KAGGLE_COMPETITION),
            message=args.message or "RL full",
        )
        cmd_submit(ns_submit)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RL pipeline: train policy, solve test, evaluate, submit. Goal: minimal total score.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=False, help="команда (по умолчанию: полный цикл train→solve→evaluate→submit)")

    # train
    p = sub.add_parser("train", help="Обучить политики (BC, опционально PG) для n из теста или n_list")
    p.add_argument("--test", default=_DEFAULT_TEST_PATH, help="CSV с id, permutation (для определения n)")
    p.add_argument("--out-dir", default=_DEFAULT_MODELS_DIR, help="Каталог для сохранения policy_n_{n}.pt")
    p.add_argument("--n-list", type=int, nargs="*", default=None, help="Явный список n (если нет теста)")
    p.add_argument("--trajectories", type=int, default=5000, help="Число траекторий на n для BC")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pg-epochs", type=int, default=0, help="Доп. эпохи policy gradient после BC")
    p.add_argument("--pg-rollouts", type=int, default=500)
    p.add_argument("--pg-lr", type=float, default=1e-4)
    p.set_defaults(run=cmd_train)

    # solve
    p = sub.add_parser("solve", help="Решить тест RL (или baseline при отсутствии модели), записать CSV")
    p.add_argument("--test", default=_DEFAULT_TEST_PATH)
    p.add_argument("--models", default=_DEFAULT_MODELS_DIR, help="Каталог с policy_n_{n}.pt")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda / cpu (по умолчанию авто)")
    p.set_defaults(run=cmd_solve)

    # analyze
    p = sub.add_parser("analyze", help="Сводка по логу экспериментов (runs/experiment_results.jsonl)")
    p.add_argument("--max-entries", type=int, default=50, help="Сколько последних строк лога учитывать")
    p.set_defaults(run=cmd_analyze)

    # evaluate
    p = sub.add_parser("evaluate", help="Сравнить submission с baseline")
    p.add_argument("--test", default=_DEFAULT_TEST_PATH)
    p.add_argument("--submission", required=True)
    p.set_defaults(run=cmd_evaluate)

    # submit
    p = sub.add_parser("submit", help="Отправить файл на Kaggle")
    p.add_argument("--file", default="submission.csv")
    p.add_argument("--competition", "-c", default=os.environ.get("KAGGLE_COMPETITION", _DEFAULT_KAGGLE_COMPETITION))
    p.add_argument("--message", "-m", default="")
    p.set_defaults(run=cmd_submit)

    # full: optional train -> solve -> optional evaluate -> optional submit
    p = sub.add_parser("full", help="train (если --train) -> solve -> evaluate/submit. Один запуск для минимального скора.")
    p.add_argument("--train", action="store_true", help="Сначала обучить политики (BC+PG), затем solve")
    p.add_argument("--test", default=_DEFAULT_TEST_PATH)
    p.add_argument("--models", default=_DEFAULT_MODELS_DIR, help="Каталог моделей (и для train, и для solve)")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--trajectories", type=int, default=5000, help="Для --train: траекторий на n")
    p.add_argument("--epochs", type=int, default=30, help="Для --train: эпохи BC")
    p.add_argument("--pg-epochs", type=int, default=0, help="Для --train: эпохи PG после BC")
    p.add_argument("--pg-rollouts", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pg-lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--evaluate", action="store_true", help="После solve вызвать evaluate")
    p.add_argument("--submit", action="store_true", help="После solve отправить на Kaggle")
    p.add_argument("--competition", "-c", default=os.environ.get("KAGGLE_COMPETITION", _DEFAULT_KAGGLE_COMPETITION))
    p.add_argument("--message", "-m", default="")
    p.set_defaults(run=cmd_full)

    args = parser.parse_args()
    if args.command is None:
        # Запуск без аргументов: полный цикл train -> solve -> evaluate -> submit
        args.command = "full"
        args.run = cmd_full
        args.train = True
        args.evaluate = True
        args.submit = True
        args.test = _DEFAULT_TEST_PATH
        args.models = _DEFAULT_MODELS_DIR
        args.out = "submission.csv"
        args.limit = None
        args.device = None
        args.trajectories = 5000
        args.epochs = 30
        args.pg_epochs = 0
        args.pg_rollouts = 500
        args.batch_size = 256
        args.lr = 1e-3
        args.pg_lr = 1e-4
        args.seed = 42
        args.competition = os.environ.get("KAGGLE_COMPETITION", _DEFAULT_KAGGLE_COMPETITION)
        args.message = ""
    args.run(args)


if __name__ == "__main__":
    main()
