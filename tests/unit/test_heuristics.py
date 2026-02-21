# -*- coding: utf-8 -*-
"""Unit-тесты: эвристики и beam."""

import pytest

from src.core import apply_moves, pancake_sort_moves
from src.heuristics import gap_h, breakpoints2, mix_h, make_h, beam_improve_or_baseline_h


class TestGapH:
    def test_identity(self):
        assert gap_h([0, 1, 2]) == 0

    def test_one_gap(self):
        # [1,0,2] — разрывы между -1,1 и 0,2 и 2,3
        assert gap_h([1, 0, 2]) >= 1


class TestBreakpoints2:
    def test_identity(self):
        assert breakpoints2([0, 1, 2]) == 0

    def test_first_not_zero(self):
        assert breakpoints2([1, 0, 2]) >= 1


class TestMixH:
    def test_call(self):
        v = mix_h([1, 0, 2], alpha=0.5)
        assert isinstance(v, float)


class TestMakeH:
    def test_alpha_zero(self):
        h = make_h(0.0)
        assert h([0, 1, 2]) == 0.0

    def test_alpha_nonzero(self):
        h = make_h(0.5)
        assert isinstance(h([1, 0, 2]), float)


class TestBeamImprove:
    def test_returns_valid_solution(self):
        perm = [1, 0, 2]
        moves = beam_improve_or_baseline_h(
            perm,
            baseline_moves_fn=pancake_sort_moves,
            h_fn=make_h(0.0),
            beam_width=8,
            depth=6,
            w=0.5,
        )
        assert apply_moves(perm, moves) == [0, 1, 2]

    def test_length_not_worse_than_baseline(self):
        perm = [2, 0, 1]
        base = pancake_sort_moves(perm)
        moves = beam_improve_or_baseline_h(
            perm,
            baseline_moves_fn=pancake_sort_moves,
            h_fn=make_h(0.0),
            beam_width=16,
            depth=10,
            w=0.5,
        )
        assert len(moves) <= len(base)
