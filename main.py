#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Точка входа для запуска и оценки решений Pancake Problem.

Примеры:
  python main.py
  python main.py solve --method baseline --test test.csv --out submission.csv
  python main.py solve --method beam --test test.csv --out submission_beam.csv
  python main.py solve --method notebook --solver v3_1 --test test.csv --out submission_nb.csv
  python main.py evaluate --test test.csv --submission submission.csv
  python main.py evaluate --test test.csv --submission baseline/submission.csv
  python main.py check-steps --submission baseline/sample_submission.csv
  python main.py compare --best baseline/submission.csv
  python main.py merge --base baseline/submission.csv --partials submission.csv --out final.csv
  python main.py list-solvers
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Корень проекта в sys.path для импорта src
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.core import (
    parse_permutation,
    pancake_sort_moves,
    moves_to_str,
)
from src.heuristics import beam_improve_or_baseline_h, make_h
from src.notebook_search import (
    get_solver,
    SOLVER_REGISTRY,
    pancake_sort_v3_1,
    pancake_sort_v3_5,
    pancake_sort_v4,
)
from src.submission import (
    evaluate_submission_vs_baseline,
    check_steps,
    compare,
    merge_submissions_with_partials,
    process_row,
)
from src.crossings import (
    solve_notebook_then_beam,
    solve_baseline_then_beam,
    solve_unified,
)


def _ensure_n(df: pd.DataFrame) -> pd.DataFrame:
    if "n" not in df.columns and "permutation" in df.columns:
        df = df.copy()
        df["n"] = df["permutation"].apply(lambda x: len(parse_permutation(x)))
    return df


# Источник перестановок по умолчанию (для всех запусков и для Kaggle)
_DEFAULT_TEST_PATH = "baseline/sample_submission.csv"
# Соревнование Kaggle (можно переопределить через --competition или KAGGLE_COMPETITION)
_DEFAULT_KAGGLE_COMPETITION = "CayleyPy-pancake"

# Встроенный пример теста (если файл теста не найден)
_DEFAULT_TEST_CSV = """id,permutation
1,"1,0,2"
2,"0,2,1"
3,"2,1,0"
"""


def cmd_solve(args: argparse.Namespace) -> None:
    """Решить тест выбранным методом и сохранить submission CSV (id, solution) для Kaggle."""
    test_path = Path(args.test)
    if not test_path.exists():
        default_path = Path(_DEFAULT_TEST_PATH)
        if default_path.exists():
            test_path = default_path
            print(f"Файл {args.test} не найден — используем {_DEFAULT_TEST_PATH}.")
    if not test_path.exists():
        import io
        print(f"Файл {args.test} не найден — используем встроенный пример (3 строки).")
        test_df = pd.read_csv(io.StringIO(_DEFAULT_TEST_CSV))
    else:
        test_df = pd.read_csv(test_path)
        if str(test_path) == _DEFAULT_TEST_PATH:
            print(f"Перестановки из {_DEFAULT_TEST_PATH} ({len(test_df)} строк).")
    test_df = _ensure_n(test_df)

    rows = []
    for i, row in test_df.iterrows():
        rid = int(row["id"])
        perm = parse_permutation(row["permutation"])
        if args.limit and len(rows) >= args.limit:
            break
        if args.max_n and len(perm) > args.max_n:
            rows.append({"id": rid, "solution": moves_to_str(pancake_sort_moves(perm))})
            continue
        if args.method == "baseline":
            moves = pancake_sort_moves(perm)
        elif args.method == "beam":
            moves = solve_baseline_then_beam(
                perm,
                beam_width=args.beam_width,
                depth=args.depth,
                alpha=args.alpha,
                w=args.w,
            )
        elif args.method == "notebook":
            func, default_t = get_solver(args.solver)
            t = getattr(args, "treshold", None) or default_t
            result = func(perm, t)
            moves = list(result[0][0]) if isinstance(result[0], tuple) else list(result[0])
        elif args.method == "crossing-notebook-beam":
            moves = solve_notebook_then_beam(
                perm,
                treshold=getattr(args, "treshold", 3),
                beam_width=args.beam_width,
                depth=args.depth,
                alpha=args.alpha,
                w=args.w,
            )
        elif args.method == "unified":
            moves = solve_unified(
                perm,
                use_notebook_baseline=args.notebook_baseline,
                treshold=getattr(args, "treshold", 3),
                beam_width=args.beam_width,
                depth=args.depth,
                alpha=args.alpha,
                w=args.w,
            )
        else:
            raise SystemExit(f"Unknown method: {args.method}")
        rows.append({"id": rid, "solution": moves_to_str(moves)})
        if (len(rows)) % 100 == 0 and len(rows) > 0:
            print(f"  solved {len(rows)} ...", flush=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    total = out_df["solution"].fillna("").apply(lambda s: s.count(".") + 1 if isinstance(s, str) and s.strip() else 0).sum()
    print(f"Saved {len(out_df)} rows to {args.out} | total moves (score): {int(total)}")

    if getattr(args, "submit", False):
        comp = getattr(args, "competition", None) or os.environ.get("KAGGLE_COMPETITION") or _DEFAULT_KAGGLE_COMPETITION
        _do_kaggle_submit(args.out, comp, getattr(args, "message", "") or f"{args.method} {len(out_df)} rows")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Сравнить submission с baseline по тесту."""
    test_df = pd.read_csv(args.test)
    sub_df = pd.read_csv(args.submission)
    test_df = _ensure_n(test_df)
    stats = evaluate_submission_vs_baseline(
        test_df,
        sub_df,
        baseline_moves_fn=pancake_sort_moves,
        log_every=args.log_every,
        save_detailed_path=args.details,
    )
    print("--- Evaluation vs baseline ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if args.details:
        print(f"  details saved to {args.details}")


def cmd_check_steps(args: argparse.Namespace) -> None:
    """Проверить корректность решений в submission (нужен test для permutation)."""
    sub_df = pd.read_csv(args.submission)
    if "permutation" not in sub_df.columns and args.test:
        test_df = pd.read_csv(args.test)[["id", "permutation"]]
        sub_df = sub_df.merge(test_df, on="id", how="left")
    wrong = check_steps(sub_df)
    print(f"Wrong solutions: {len(wrong)} ids: {wrong[:20]}{'...' if len(wrong) > 20 else ''}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Сводка по best_df по n (score, prob_step, potential)."""
    best_df = pd.read_csv(args.best)
    if "score" not in best_df.columns and "solution" in best_df.columns:
        best_df = best_df.copy()
        best_df["score"] = best_df["solution"].fillna("").apply(lambda s: s.count(".") + 1 if isinstance(s, str) and s.strip() else 0)
    compare(best_df, n_list=args.n_list)


def cmd_merge(args: argparse.Namespace) -> None:
    """Объединить base и partial сабмиты в один файл."""
    merge_submissions_with_partials(
        base_paths=args.base,
        partial_paths=args.partials or [],
        out_path=args.out,
        save_source_column=args.source_column,
        tie_break=args.tie_break,
    )


def _kaggle_submit_http(file_path: Path, competition: str, message: str, config_dirs: list[Path]) -> None:
    """Отправка на Kaggle через HTTP API (обход сломанного CLI/kagglesdk)."""
    import base64
    import json
    # #region agent log
    try:
        _log = lambda **kw: open("debug-aa773c.log", "a", encoding="utf-8").write(json.dumps({"sessionId": "aa773c", "location": "main.py:_kaggle_submit_http", "timestamp": __import__("time").time() * 1000, **kw}) + "\n")
    except Exception:
        _log = lambda **kw: None
    # #endregion
    try:
        import requests
    except ImportError:
        raise SystemExit("Для HTTP-отправки нужен requests: pip install requests")
    kaggle_json_path = None
    for d in config_dirs:
        p = d / "kaggle.json"
        if p.exists():
            kaggle_json_path = p
            break
    _log(message="kaggle_json_resolved", data={"found": kaggle_json_path is not None, "config_dirs": [str(x) for x in config_dirs]}, hypothesisId="H1")
    if not kaggle_json_path:
        raise SystemExit(
            "Не найден kaggle.json. Положи его в корень проекта или в ~/.kaggle/\n"
            "Скачай: https://www.kaggle.com/settings → API → Create New Token"
        )
    with open(kaggle_json_path, encoding="utf-8") as f:
        creds = json.load(f)
    username = creds.get("username") or creds.get("username_")
    key = creds.get("key") or creds.get("key_")
    if not username or not key:
        raise SystemExit("В kaggle.json должны быть поля username и key")
    auth = base64.b64encode(f"{username}:{key}".encode()).decode()
    url = f"https://www.kaggle.com/api/v1/competitions/submit/{competition}"
    with open(file_path, "rb") as f:
        files = {"submission": (file_path.name, f, "text/csv")}
        data = {"description": (None, message or file_path.name)}
        r = requests.post(
            url,
            files=files,
            data=data,
            headers={"Authorization": f"Basic {auth}"},
            timeout=120,
        )
    _log(message="http_response", data={"status_code": r.status_code, "text_preview": (r.text or "")[:300]}, hypothesisId="H2")
    if r.status_code == 200:
        print("Сабмит отправлен (HTTP). Результат на странице соревнования на Kaggle.")
        return
    if r.status_code == 404:
        raise SystemExit(
            "Kaggle больше не принимает сабмиты по старому HTTP API (404).\n"
            "Исправь CLI и отправляй через него: pip install \"kaggle>=1.5.0,<1.8.0\"\n"
            "После установки снова запусти с флагом --submit."
        )
    raise SystemExit(f"Kaggle API ответил {r.status_code}: {r.text[:500]}")


def _do_kaggle_submit(file_path: str, competition: str | None, message: str) -> None:
    """Общая логика отправки файла на Kaggle (сначала CLI, при ошибке kagglesdk — HTTP)."""
    import shutil
    import subprocess
    if not competition:
        raise SystemExit(
            "Укажи соревнование: --competition <slug> или задай KAGGLE_COMPETITION.\n"
            "Слаг смотри в URL: https://www.kaggle.com/c/<competition>"
        )
    path = Path(file_path).resolve()
    if not path.exists():
        raise SystemExit(f"Файл не найден: {path}")
    project_root = Path(__file__).resolve().parent
    config_dirs = [project_root]
    if os.environ.get("KAGGLE_CONFIG_DIR"):
        config_dirs.insert(0, Path(os.environ["KAGGLE_CONFIG_DIR"]))
    config_dirs.append(Path.home() / ".kaggle")
    kaggle_exe = shutil.which("kaggle")
    if not kaggle_exe and os.name == "nt":
        scripts = Path(sys.executable).resolve().parent / "Scripts" / "kaggle.exe"
        if scripts.exists():
            kaggle_exe = str(scripts)
    if not kaggle_exe:
        kaggle_exe = None
    cmd = (
        [kaggle_exe, "competitions", "submit", "-c", competition, "-f", str(path), "-m", message or path.name]
        if kaggle_exe
        else [sys.executable, "-m", "kaggle", "competitions", "submit", "-c", competition, "-f", str(path), "-m", message or path.name]
    )
    env = os.environ.copy()
    if (project_root / "kaggle.json").exists():
        env["KAGGLE_CONFIG_DIR"] = str(project_root)
    print("Отправляю на Kaggle:", " ".join(cmd))
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise subprocess.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)
        print("Сабмит отправлен. Результат на странице соревнования на Kaggle.")
    except FileNotFoundError:
        raise SystemExit(
            "Установи Kaggle API: pip install kaggle\n"
            "Настрой токен: скачай kaggle.json из https://www.kaggle.com/settings (API → Create New Token),\n"
            "положи в ~/.kaggle/kaggle.json (Windows: %USERPROFILE%\\.kaggle\\kaggle.json)"
        )
    except subprocess.CalledProcessError as e:
        err_text = (getattr(e, "stderr", None) or getattr(e, "stdout", None) or "") or ""
        if isinstance(err_text, bytes):
            err_text = err_text.decode("utf-8", errors="replace")
        if "kagglesdk" in err_text or "get_access_token_from_env" in err_text:
            # #region agent log
            open("debug-aa773c.log", "a", encoding="utf-8").write(__import__("json").dumps({"sessionId": "aa773c", "location": "main.py:do_kaggle", "message": "cli_failed_using_http_fallback", "data": {"stderr_snippet": err_text[:200]}, "timestamp": __import__("time").time() * 1000, "hypothesisId": "H0"}) + "\n")
            # #endregion
            print("CLI не сработал (kagglesdk), пробую отправить через HTTP API…")
            try:
                _kaggle_submit_http(path, competition, message or path.name, config_dirs)
            except SystemExit:
                raise
            except Exception as exc:
                # #region agent log
                open("debug-aa773c.log", "a", encoding="utf-8").write(__import__("json").dumps({"sessionId": "aa773c", "location": "main.py:do_kaggle", "message": "http_exception", "data": {"type": type(exc).__name__, "msg": str(exc)}, "timestamp": __import__("time").time() * 1000, "hypothesisId": "H3"}) + "\n")
                # #endregion
                raise SystemExit(
                    f"HTTP-отправка не удалась: {exc}\n"
                    "Обновление (pip install -U kaggle) даёт несовместимость с kagglesdk.\n"
                    "Поставь старую версию: pip install \"kaggle>=1.5.0,<1.8.0\""
                )
            return
        raise SystemExit(f"Ошибка Kaggle API: {e}\n{err_text}")


def cmd_submit(args: argparse.Namespace) -> None:
    """Отправить submission.csv на Kaggle через API (нужен kaggle и kaggle.json в ~/.kaggle/)."""
    _do_kaggle_submit(
        args.file,
        args.competition,
        args.message or f"submit {Path(args.file).name}",
    )


def cmd_list_solvers(args: argparse.Namespace) -> None:
    """Показать доступные солверы блокнота."""
    print("Available notebook solvers (for --method notebook --solver <name>):")
    for name, (fn, default_t) in SOLVER_REGISTRY.items():
        print(f"  {name}  (default treshold={default_t})")


def cmd_process_row(args: argparse.Namespace) -> None:
    """Прогнать одну или несколько строк через солвер блокнота (для отладки)."""
    test_df = pd.read_csv(args.test)
    solver_name = args.solver or "v3_1"
    func, default_t = get_solver(solver_name)
    t = args.treshold or default_t
    rows = []
    for i, row in test_df.iterrows():
        if args.limit and len(rows) >= args.limit:
            break
        out = process_row(
            row.to_dict(),
            func,
            treshold=t,
            save=False,
            from_target=args.from_target,
        )
        rows.append(out)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Saved {len(out_df)} rows to {args.out} | score sum: {out_df['score'].sum()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pancake Problem: запуск решателей и оценка сабмитов.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=False)

    # solve
    p = sub.add_parser("solve", help="Решить тест и сохранить submission CSV (для Kaggle)")
    p.add_argument("--test", default=_DEFAULT_TEST_PATH, help="CSV с id, permutation (по умолчанию: baseline/sample_submission.csv)")
    p.add_argument("--out", default="submission.csv", help="Выходной CSV (id, solution) для загрузки на Kaggle")
    p.add_argument("--method", choices=["baseline", "beam", "notebook", "crossing-notebook-beam", "unified"], default="baseline")
    p.add_argument("--solver", default="v3_1", help="Солвер блокнота при method=notebook (v3_1, v3_5, v4)")
    p.add_argument("--treshold", type=int, default=None, help="Порог для солверов блокнота")
    p.add_argument("--notebook-baseline", action="store_true", help="Для unified: использовать блокнот как baseline")
    p.add_argument("--beam-width", type=int, default=128)
    p.add_argument("--depth", type=int, default=128)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--w", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None, help="Макс. число строк (для теста)")
    p.add_argument("--max-n", type=int, default=None, help="Для perm с n>max_n использовать только baseline")
    p.add_argument("--submit", action="store_true", help="После решения отправить --out на Kaggle")
    p.add_argument("--competition", "-c", default=os.environ.get("KAGGLE_COMPETITION", _DEFAULT_KAGGLE_COMPETITION), help="Слаг соревнования (по умолчанию: CayleyPy-pancake)")
    p.add_argument("--message", "-m", default="", help="Сообщение к сабмиту (для --submit)")
    p.set_defaults(run=cmd_solve)

    # evaluate
    p = sub.add_parser("evaluate", help="Сравнить submission с baseline")
    p.add_argument("--test", default="test.csv")
    p.add_argument("--submission", required=True, help="Submission CSV")
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--details", default=None, help="Путь для сохранения детальной таблицы gain по id")
    p.set_defaults(run=cmd_evaluate)

    # check-steps
    p = sub.add_parser("check-steps", help="Проверить корректность решений")
    p.add_argument("--submission", required=True)
    p.add_argument("--test", default=None, help="Если в submission нет permutation")
    p.set_defaults(run=cmd_check_steps)

    # compare
    p = sub.add_parser("compare", help="Сводка по best_df по n")
    p.add_argument("--best", required=True, help="CSV с колонками id, solution, score (или n)")
    p.add_argument("--n-list", type=int, nargs="*", default=[5, 12, 15, 16, 20, 25, 30, 35, 40, 45, 50, 75, 100])
    p.set_defaults(run=cmd_compare)

    # merge
    p = sub.add_parser("merge", help="Объединить base + partial сабмиты")
    p.add_argument("--base", nargs="+", required=True)
    p.add_argument("--partials", nargs="*", default=[])
    p.add_argument("--out", default="submission_final.csv")
    p.add_argument("--source-column", action="store_true", default=True)
    p.add_argument("--tie-break", choices=["keep_base", "prefer_partial"], default="keep_base")
    p.set_defaults(run=cmd_merge)

    # submit (Kaggle API)
    p = sub.add_parser("submit", help="Отправить submission на Kaggle (pip install kaggle, настроить kaggle.json)")
    p.add_argument("--file", default="submission.csv", help="Файл с колонками id, solution")
    p.add_argument("--competition", "-c", default=os.environ.get("KAGGLE_COMPETITION", _DEFAULT_KAGGLE_COMPETITION), help="Слаг соревнования (по умолчанию: CayleyPy-pancake)")
    p.add_argument("--message", "-m", default="", help="Сообщение к сабмиту")
    p.set_defaults(run=cmd_submit)

    # list-solvers
    p = sub.add_parser("list-solvers", help="Список солверов блокнота")
    p.set_defaults(run=cmd_list_solvers)

    # process-row (notebook pipeline on test rows)
    p = sub.add_parser("process-row", help="Прогнать тест через солвер блокнота (process_row)")
    p.add_argument("--test", default="test.csv")
    p.add_argument("--out", default="submission_process.csv")
    p.add_argument("--solver", default="v3_1")
    p.add_argument("--treshold", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--from-target", action="store_true")
    p.set_defaults(run=cmd_process_row)

    args = parser.parse_args()
    if args.command is None:
        # Команда по умолчанию: solve baseline на перестановках из baseline/
        args = argparse.Namespace(
            command="solve",
            run=cmd_solve,
            test=_DEFAULT_TEST_PATH,
            out="submission.csv",
            method="baseline",
            solver="v3_1",
            treshold=None,
            notebook_baseline=False,
            beam_width=128,
            depth=128,
            alpha=0.0,
            w=0.5,
            limit=None,
            max_n=None,
            submit=True,
            competition=os.environ.get("KAGGLE_COMPETITION", _DEFAULT_KAGGLE_COMPETITION),
            message="",
        )
    args.run(args)


if __name__ == "__main__":
    main()
