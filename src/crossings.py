# -*- coding: utf-8 -*-
"""Скрещивания: блокнот + beam, блокнот + 91584, все три вместе."""

from __future__ import annotations

from typing import List, Iterable, Callable

from .core import (
    parse_permutation,
    pancake_sort_moves,
    apply_moves,
    moves_to_str,
)
from .heuristics import beam_improve_or_baseline_h, make_h
from .notebook_search import notebook_baseline_v3_1


def solve_notebook_then_beam(
    perm: Iterable[int],
    *,
    treshold: int = 3,
    beam_width: int = 128,
    depth: int = 128,
    alpha: float = 0.0,
    w: float = 0.5,
) -> List[int]:
    """Скрещивание 2: базовое решение из блокнота (v3_1), затем улучшение beam search."""
    baseline_fn: Callable[[Iterable[int]], List[int]] = lambda p: notebook_baseline_v3_1(
        p, treshold=treshold
    )
    return beam_improve_or_baseline_h(
        perm,
        baseline_moves_fn=baseline_fn,
        h_fn=make_h(alpha),
        beam_width=beam_width,
        depth=depth,
        w=w,
        log=False,
    )


def solve_baseline_then_beam(
    perm: Iterable[int],
    *,
    beam_width: int = 128,
    depth: int = 128,
    alpha: float = 0.0,
    w: float = 0.5,
) -> List[int]:
    """Классический 91584-пайплайн: baseline = pancake_sort_moves, затем beam."""
    return beam_improve_or_baseline_h(
        perm,
        baseline_moves_fn=pancake_sort_moves,
        h_fn=make_h(alpha),
        beam_width=beam_width,
        depth=depth,
        w=w,
        log=False,
    )


def solve_unified(
    perm: Iterable[int],
    *,
    use_notebook_baseline: bool = True,
    treshold: int = 3,
    beam_width: int = 128,
    depth: int = 128,
    alpha: float = 0.0,
    w: float = 0.5,
) -> List[int]:
    """Скрещивание 3/4: единый вход — либо блокнот-baseline, либо классический; затем beam."""
    if use_notebook_baseline:
        baseline_fn: Callable[[Iterable[int]], List[int]] = lambda p: notebook_baseline_v3_1(
            p, treshold=treshold
        )
    else:
        baseline_fn = pancake_sort_moves

    return beam_improve_or_baseline_h(
        perm,
        baseline_moves_fn=baseline_fn,
        h_fn=make_h(alpha),
        beam_width=beam_width,
        depth=depth,
        w=w,
        log=False,
    )
