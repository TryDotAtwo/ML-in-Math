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
from .notebook_search import notebook_baseline_v3_1, notebook_baseline_v4, get_solver


def solve_notebook_then_beam(
    perm: Iterable[int],
    *,
    notebook_solver: str = "v3_1",
    treshold: int | None = None,
    notebook_max_states: int | None = None,
    beam_width: int = 128,
    depth: int = 128,
    alpha: float = 0.0,
    w: float = 0.5,
) -> List[int]:
    """Скрещивание 2: базовое решение из блокнота (v3_1 или v4), затем улучшение beam search."""
    _, default_t = get_solver(notebook_solver)
    t = treshold if treshold is not None else default_t
    if notebook_solver == "v4":
        baseline_fn: Callable[[Iterable[int]], List[int]] = lambda p: notebook_baseline_v4(
            p, treshold=t
        )
    else:
        baseline_fn = lambda p: notebook_baseline_v3_1(
            p, treshold=t, max_states=notebook_max_states
        )
    return beam_improve_or_baseline_h(
        perm,
        baseline_moves_fn=baseline_fn,
        h_fn=make_h(alpha),
        beam_width=beam_width,
        depth=depth,
        w=w,
        log=False,
        prune_best_g_each_layer=True,
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
        prune_best_g_each_layer=True,
    )


def solve_unified(
    perm: Iterable[int],
    *,
    use_notebook_baseline: bool = True,
    notebook_solver: str = "v3_1",
    treshold: int | None = None,
    notebook_max_states: int | None = None,
    beam_width: int = 128,
    depth: int = 128,
    alpha: float = 0.0,
    w: float = 0.5,
) -> List[int]:
    """Скрещивание 3/4: единый вход — либо блокнот-baseline (v3_1/v4), либо классический; затем beam."""
    if use_notebook_baseline:
        _, default_t = get_solver(notebook_solver)
        t = treshold if treshold is not None else default_t
        if notebook_solver == "v4":
            baseline_fn: Callable[[Iterable[int]], List[int]] = lambda p: notebook_baseline_v4(
                p, treshold=t
            )
        else:
            baseline_fn = lambda p: notebook_baseline_v3_1(
                p, treshold=t, max_states=notebook_max_states
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
        prune_best_g_each_layer=True,
    )
