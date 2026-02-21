# -*- coding: utf-8 -*-
"""Unit-тесты: core (permutation, moves, baseline)."""

import pytest

from src.core import (
    parse_permutation,
    moves_to_str,
    moves_len,
    solution_to_moves,
    apply_move_copy,
    apply_moves,
    is_solved,
    pancake_sort_moves,
)


class TestParsePermutation:
    def test_empty(self):
        assert parse_permutation("") == []
        assert parse_permutation(None) == []

    def test_simple(self):
        assert parse_permutation("1,2,0") == [1, 2, 0]
        assert parse_permutation("0,1,2") == [0, 1, 2]

    def test_spaces(self):
        assert parse_permutation(" 0 , 1 , 2 ") == [0, 1, 2]


class TestMovesStr:
    def test_empty(self):
        assert moves_to_str([]) == ""

    def test_simple(self):
        assert moves_to_str([2, 5, 3]) == "R2.R5.R3"


class TestMovesLen:
    def test_empty(self):
        assert moves_len(None) == 0
        assert moves_len("") == 0

    def test_string(self):
        assert moves_len("R2.R5") == 2
        assert moves_len("R3") == 1

    def test_list(self):
        assert moves_len([2, 5, 3]) == 3


class TestSolutionToMoves:
    def test_empty(self):
        assert solution_to_moves(None) == []
        assert solution_to_moves("") == []

    def test_simple(self):
        assert solution_to_moves("R2.R5.R3") == [2, 5, 3]
        assert solution_to_moves("R2") == [2]


class TestApplyMove:
    def test_identity(self):
        assert apply_move_copy([0, 1, 2], 2) == [1, 0, 2]

    def test_full_flip(self):
        assert apply_move_copy([2, 1, 0], 3) == [0, 1, 2]


class TestApplyMoves:
    def test_empty_moves(self):
        perm = [1, 0, 2]
        assert apply_moves(perm, []) == perm

    def test_solution(self):
        perm = [1, 0, 2]
        moves = [2, 2]  # R2 twice -> identity for n=3
        got = apply_moves(perm, moves)
        assert got == [0, 1, 2]


class TestIsSolved:
    def test_identity(self):
        assert is_solved([0, 1, 2]) is True
        assert is_solved([0]) is True

    def test_not_solved(self):
        assert is_solved([1, 0, 2]) is False


class TestPancakeSortMoves:
    def test_empty_single(self):
        assert pancake_sort_moves([]) == []
        assert pancake_sort_moves([0]) == []

    def test_identity(self):
        assert pancake_sort_moves([0, 1, 2]) == []

    def test_correctness(self):
        perm = [1, 0, 2]
        moves = pancake_sort_moves(perm)
        assert apply_moves(perm, moves) == [0, 1, 2]

    def test_small_n(self):
        for perm in [[2, 0, 1], [1, 2, 0]]:
            moves = pancake_sort_moves(perm)
            assert apply_moves(perm, moves) == list(range(len(perm))), perm
