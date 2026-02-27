# -*- coding: utf-8 -*-
"""Beam search с эвристикой для улучшения базового решения."""

from __future__ import annotations

from typing import List, Iterable, Tuple, Dict, Optional, Callable
from heapq import nsmallest

from ..core.moves import apply_move_copy, is_solved


def _log_print(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, flush=True)


def beam_improve_or_baseline_h(
    perm: Iterable[int],
    *,
    baseline_moves_fn: Callable[[Iterable[int]], List[int]],
    h_fn: Callable[[List[int]], float],
    beam_width: int = 8,
    depth: int = 12,
    w: float = 1.0,
    log: bool = False,
    log_every_layer: int = 1,
    prune_best_g_each_layer: bool = False,
) -> List[int]:
    """Улучшает базовое решение ограниченным beam search с оценкой f = g + w*h.
    Возвращает улучшенный список ходов или исходный baseline.
    prune_best_g_each_layer=True ограничивает память (только состояния текущего луча).
    """
    start = list(perm)

    base_moves = baseline_moves_fn(start)
    best_len = len(base_moves)
    if best_len <= 1:
        return base_moves

    k_values = range(2, len(start) + 1)

    beam: List[Tuple[float, int, List[int], List[int]]] = [
        (w * float(h_fn(start)), 0, start, [])
    ]

    best_path: Optional[List[int]] = None
    best_g: Dict[Tuple[int, ...], int] = {tuple(start): 0}

    _log_print(log, f"[beam] start n={len(start)} base_len={best_len} bw={beam_width} depth={depth} w={w}")

    for layer in range(1, depth + 1):
        candidates: List[Tuple[float, int, List[int], List[int]]] = []
        improved_this_layer = 0
        for f, g, state, path in beam:
            if g >= best_len:
                continue

            for k in k_values:
                new_g = g + 1
                if new_g >= best_len:
                    continue

                nxt = apply_move_copy(state, k)
                key = tuple(nxt)

                prevg = best_g.get(key)
                if prevg is not None and prevg <= new_g:
                    continue
                best_g[key] = new_g

                if is_solved(nxt):
                    best_len = new_g
                    best_path = path + [k]
                    improved_this_layer += 1
                    continue

                h = float(h_fn(nxt))
                new_f = new_g + w * h
                if new_f < best_len:
                    candidates.append((new_f, new_g, nxt, path + [k]))

        if log and (layer % max(1, log_every_layer) == 0):
            _log_print(
                True,
                f"[beam] layer={layer:03d} beam_in={len(beam)} cand={len(candidates)} "
                f"improved={improved_this_layer} best_len={best_len}"
            )

        if not candidates:
            break

        beam = nsmallest(beam_width, candidates, key=lambda x: x[0])
        if prune_best_g_each_layer:
            best_g = {tuple(state): g for (_, g, state, _) in beam}
        if best_len <= 2:
            break

    return best_path if best_path is not None else base_moves
