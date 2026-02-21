# -*- coding: utf-8 -*-
"""Unit-тесты: солверы блокнота (v3_1)."""

import pytest

from src.core import apply_moves
from src.notebook_search import pancake_sort_v3_1, notebook_baseline_v3_1


class TestPancakeSortV3_1:
    def test_returns_tuple(self):
        perm = [1, 0, 2]
        result = pancake_sort_v3_1(perm, treshold=3)
        assert len(result) == 4
        moves_tuple, search, max_len, total_iter = result
        assert isinstance(moves_tuple, tuple)
        moves = list(moves_tuple)
        assert apply_moves(perm, moves) == [0, 1, 2]

    def test_small_n5(self):
        perm = [2, 0, 1, 4, 3]
        result = pancake_sort_v3_1(perm, treshold=5)
        moves = list(result[0][0])
        assert apply_moves(perm, moves) == list(range(5))


class TestNotebookBaselineV3_1:
    def test_returns_list(self):
        perm = [1, 0, 2]
        moves = notebook_baseline_v3_1(perm, treshold=3)
        assert isinstance(moves, list)
        assert all(isinstance(k, int) for k in moves)
        assert apply_moves(perm, moves) == [0, 1, 2]
