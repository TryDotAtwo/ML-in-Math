# -*- coding: utf-8 -*-
"""Интеграционные тесты: пайплайн и форматы."""

import pandas as pd
import pytest

from src.core import (
    parse_permutation,
    pancake_sort_moves,
    apply_moves,
    moves_to_str,
    solution_to_moves,
)
from src.submission import evaluate_submission_vs_baseline
from src.crossings import solve_notebook_then_beam, solve_baseline_then_beam, solve_unified


def _make_test_df():
    rows = [
        {"id": 1, "permutation": "1,0,2"},
        {"id": 2, "permutation": "0,2,1"},
    ]
    return pd.DataFrame(rows)


class TestFormats:
    def test_roundtrip(self):
        moves = [2, 5, 3]
        s = moves_to_str(moves)
        back = solution_to_moves(s)
        assert back == moves

    def test_apply_from_string(self):
        perm = [1, 0, 2]
        moves = pancake_sort_moves(perm)
        s = moves_to_str(moves)
        back = solution_to_moves(s)
        assert apply_moves(perm, back) == [0, 1, 2]


class TestEvaluateSubmission:
    def test_vs_baseline(self):
        test_df = _make_test_df()
        sub_rows = []
        for _, row in test_df.iterrows():
            perm = parse_permutation(row["permutation"])
            moves = pancake_sort_moves(perm)
            sub_rows.append({
                "id": row["id"],
                "solution": moves_to_str(moves),
            })
        sub_df = pd.DataFrame(sub_rows)
        stats = evaluate_submission_vs_baseline(
            test_df, sub_df, baseline_moves_fn=pancake_sort_moves
        )
        assert stats["worse_cases"] == 0
        assert stats["submission_total"] == stats["baseline_total"]


class TestCheckSteps:
    def test_all_correct(self):
        from src.submission import check_steps

        df = pd.DataFrame([
            {"id": 1, "permutation": "1,0,2", "solution": "R2.R2"},
            {"id": 2, "permutation": "0,2,1", "solution": "R3.R2"},
        ])
        wrong = check_steps(df)
        assert wrong == []

    def test_wrong_solution(self):
        from src.submission import check_steps

        # [0,2,1] with R2 only -> [2,0,1], not identity
        df = pd.DataFrame([
            {"id": 1, "permutation": "0,2,1", "solution": "R2"},
        ])
        wrong = check_steps(df)
        assert 1 in wrong


class TestProcessRow:
    def test_process_row_v3_1(self):
        from src.submission import process_row
        from src.notebook_search import pancake_sort_v3_1

        row = {"id": 10, "permutation": "1,0,2", "n": 3}
        out = process_row(row, pancake_sort_v3_1, treshold=3, save=False)
        assert out["id"] == 10
        assert out["score"] == len(out["solution"].split("."))
        from src.core import parse_permutation, solution_to_moves, apply_moves
        final = apply_moves(parse_permutation(row["permutation"]), solution_to_moves(out["solution"]))
        assert final == [0, 1, 2]


class TestBestSolution:
    def test_best_choice(self):
        from src.submission import best_solution

        # текущий сабмит лучше по одному id
        submission_df = pd.DataFrame([
            {"id": 1, "solution": "R2.R2", "score": 2},
            {"id": 2, "solution": "R2", "score": 1},
        ])
        best_df = pd.DataFrame([
            {"id": 1, "solution": "R2.R2.R2", "score": 3},
            {"id": 2, "solution": "R2", "score": 1},
        ])
        best_out, stats = best_solution(submission_df, best_df=best_df, safe=False)
        assert stats["best"] >= 1
        assert best_out[best_out["id"] == 1]["solution"].iloc[0] == "R2.R2"

    def test_best_no_path(self):
        from src.submission import best_solution

        sub = pd.DataFrame([{"id": 1, "solution": "R2", "score": 1}])
        best_out, _ = best_solution(sub, best_df=None, best_path=None, safe=False)
        assert len(best_out) == 1


class TestCrossings:
    def test_notebook_then_beam(self):
        perm = [2, 0, 1]
        moves = solve_notebook_then_beam(
            perm, treshold=3, beam_width=16, depth=8, alpha=0.0, w=0.5
        )
        assert apply_moves(perm, moves) == list(range(len(perm)))

    def test_baseline_then_beam(self):
        perm = [2, 0, 1]
        moves = solve_baseline_then_beam(perm, beam_width=16, depth=8)
        assert apply_moves(perm, moves) == list(range(len(perm)))

    def test_unified_notebook(self):
        perm = [1, 0, 2]
        moves = solve_unified(perm, use_notebook_baseline=True, treshold=3, beam_width=8, depth=6)
        assert apply_moves(perm, moves) == [0, 1, 2]

    def test_unified_baseline(self):
        perm = [1, 0, 2]
        moves = solve_unified(perm, use_notebook_baseline=False, beam_width=8, depth=6)
        assert apply_moves(perm, moves) == [0, 1, 2]
