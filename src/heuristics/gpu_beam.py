#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU-ориентированный beam search с gap-эвристикой (экспериментальный).

Идея: батчево хранить фронтир луча на GPU и для каждого шага порождать
кандидатов для всех R_k (k=2..n), считать gap_h векторизованно и выбирать
top-K по f = g + w*h. Базовое решение (classic baseline) используется как
верхняя граница и fallback.

Контракт совместим по смыслу с beam_improve_or_baseline_h:
- на вход: перестановка perm и baseline_moves_fn,
- на выход: улучшенный список ходов или baseline.

Ограничения/отличия:
- не ведётся полный best_g-словарь, поэтому возможны повторные посещения;
  полагаемся на порог best_len и эвристику.
- ориентировано на moderate n (<= 100) и beam_width до нескольких тысяч.
"""

from __future__ import annotations

from typing import Iterable, List, Callable, Optional, Dict, Tuple

import json
import time
from pathlib import Path

import torch


_DEBUG_LOG_PATH = Path("debug-4ace27.log")
_DEBUG_SESSION_ID = "4ace27"


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    """Append one NDJSON debug line to the shared debug log."""
    payload = {
        "sessionId": _DEBUG_SESSION_ID,
        "id": f"log_{int(time.time() * 1000)}",
        "timestamp": int(time.time() * 1000),
        "location": location,
        "message": message,
        "data": data,
        "runId": "gpu_beam",
        "hypothesisId": hypothesis_id,
    }
    # #region agent log
    try:
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Не мешаем основному алгоритму падать из-за логгера
        pass
    # #endregion


def _as_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _gap_h_batch(states: torch.Tensor) -> torch.Tensor:
    """Векторизованный gap_h для батча состояний.

    states: LongTensor [B, n] со значениями 0..n-1.
    Возвращает Tensor [B] с целочисленными значениями gap_h.
    """
    assert states.dim() == 2
    b, n = states.shape
    device = states.device

    # ext = [-1] + state
    minus_one = torch.full((b, 1), -1, dtype=states.dtype, device=device)
    ext = torch.cat([minus_one, states], dim=1)  # [B, n+1]

    # gaps между соседями (-1, s0), (s0,s1), ..., (s_{n-2}, s_{n-1})
    diffs = (ext[:, 1:] - ext[:, :-1]).abs() != 1  # [B, n+1-1] = [B, n]
    gaps = diffs.sum(dim=1).to(torch.int64)  # [B]

    # финальный gap между последним элементом и n
    last = states[:, -1]
    tail_gap = (last - int(n)).abs() != 1
    gaps = gaps + tail_gap.to(gaps.dtype)
    return gaps


# Кэш для индексов префиксных флипов по (n, device)
_FLIP_IDX_CACHE: Dict[Tuple[int, torch.device], torch.Tensor] = {}


def _get_flip_idx(n: int, device: torch.device) -> torch.Tensor:
    """Предвычисленные индексы для всех префиксных флипов R_k, k=2..n."""
    key = (n, device)
    cached = _FLIP_IDX_CACHE.get(key)
    if cached is not None:
        return cached
    k_list = []
    for k in range(2, n + 1):
        prefix_idx = torch.arange(k - 1, -1, -1, device=device)
        rest_idx = torch.arange(k, n, device=device)
        full_idx = torch.cat([prefix_idx, rest_idx], dim=0)  # [n]
        k_list.append(full_idx)
    flip_idx = torch.stack(k_list, dim=0)  # [num_k, n]
    _FLIP_IDX_CACHE[key] = flip_idx
    return flip_idx


@torch.no_grad()
def beam_improve_or_baseline_h_gpu(
    perm: Iterable[int],
    *,
    baseline_moves_fn: Callable[[Iterable[int]], List[int]],
    beam_width: int = 4096,
    depth: int = 4096,
    w: float = 0.5,
    device: Optional[str] = None,
    log: bool = False,
) -> List[int]:
    """Beam search на GPU с gap_h, улучшающий baseline.

    - baseline_moves_fn даёт исходный путь и верхнюю границу best_len;
    - если улучшения не найдено, возвращается baseline.
    """
    start_list = list(perm)
    n = len(start_list)
    if n <= 1:
        return []

    base_moves = baseline_moves_fn(start_list)
    best_len = len(base_moves)
    if best_len <= 1:
        return base_moves

    dev = _as_device(device)
    start = torch.tensor(start_list, dtype=torch.long, device=dev).unsqueeze(0)  # [1, n]
    target = torch.arange(n, dtype=torch.long, device=dev).unsqueeze(0)  # [1, n]

    # Инициализация луча
    beam_states = start  # [B, n]
    beam_g = torch.zeros(1, dtype=torch.int64, device=dev)  # [B]
    # beam_paths[b, t] хранит k для шага t (0..depth-1) для состояния b
    beam_paths = torch.zeros((1, depth), dtype=torch.int16, device=dev)  # [B, depth]

    # k-значения и предвычисленные индексы флипа для всех k (2..n)
    k_values = torch.arange(2, n + 1, dtype=torch.int64, device=dev)  # [n-1]
    flip_idx = _get_flip_idx(n, dev)  # [num_k, n]

    best_path: Optional[List[int]] = None

    if log:
        print(
            f"[gpu_beam] start n={n} base_len={best_len} bw={beam_width} depth={depth} w={w} device={dev}",
            flush=True,
        )

    for layer in range(1, depth + 1):
        b = beam_states.size(0)
        if b == 0:
            break

        # Быстрое отсечение "безнадёжных" родителей:
        # если g + 1 >= best_len, то новые состояния не могут улучшить baseline.
        parent_keep_mask = (beam_g + 1) < best_len
        if not bool(parent_keep_mask.any()):
            break
        keep_idx = parent_keep_mask.nonzero(as_tuple=False).squeeze(1)
        if keep_idx.numel() != b:
            beam_states = beam_states[keep_idx]
            beam_g = beam_g[keep_idx]
            beam_paths = beam_paths[keep_idx]
            b = beam_states.size(0)
            if b == 0:
                break

        # Порождаем кандидатов для всех k векторизованно.
        # beam_states: [B, n]
        num_k = k_values.numel()  # n-1

        # parent индексы и k для каждого кандидата: [B * (n-1)]
        parent_idx = torch.arange(b, device=dev).repeat_interleave(num_k)
        k_for_cand = k_values.repeat(b)

        # Генерация состояний-кандидатов:
        # расширяем beam_states до [B, num_k, n] и применяем flip_idx по dim=2.
        states_3d = beam_states.unsqueeze(1).expand(-1, num_k, -1)  # [B, num_k, n]
        idx_3d = flip_idx.unsqueeze(0).expand(b, -1, -1)  # [B, num_k, n]
        cand_3d = torch.gather(states_3d, 2, idx_3d)  # [B, num_k, n]
        candidates = cand_3d.reshape(b * num_k, n)  # [B*(n-1), n]

        new_g = beam_g[parent_idx] + 1  # [B*(n-1)]

        # Обновляем пути на GPU: cand_paths[i, :] = beam_paths[parent_idx[i], :] с k_for_cand на позиции new_g-1
        M_all = candidates.size(0)
        parent_paths = beam_paths[parent_idx]  # [M_all, depth]
        cand_paths = parent_paths.clone()
        row_idx = torch.arange(M_all, device=dev)
        step_idx = torch.clamp(new_g - 1, min=0, max=depth - 1)
        cand_paths[row_idx, step_idx] = k_for_cand.to(cand_paths.dtype)

        # Проверяем решения
        solved_mask = (candidates == target).all(dim=1)
        if solved_mask.any():
            solved_indices = solved_mask.nonzero(as_tuple=False).squeeze(1)
            for idx in solved_indices.tolist():
                g_val = int(new_g[idx].item())
                if g_val < best_len:
                    best_len = g_val
                    # Восстанавливаем путь из cand_paths для этого кандидата
                    path_tensor = cand_paths[idx, :g_val]
                    best_path = [int(x) for x in path_tensor.tolist()]

        # Кандидаты, которые не решены
        keep_mask = ~solved_mask
        if not bool(keep_mask.any()):
            break

        candidates = candidates[keep_mask]
        new_g = new_g[keep_mask]
        parent_idx = parent_idx[keep_mask]
        k_for_cand = k_for_cand[keep_mask]
        cand_paths = cand_paths[keep_mask]

        # Эвристика gap_h
        h_vals = _gap_h_batch(candidates).to(torch.float32)  # [M]
        f_vals = new_g.to(torch.float32) + w * h_vals

        # Оставляем только те, которые потенциально лучше baseline
        mask_better = f_vals < float(best_len)
        if not bool(mask_better.any()):
            # Нечего продолжать: всё хуже baseline
            break

        candidates = candidates[mask_better]
        new_g = new_g[mask_better]
        parent_idx = parent_idx[mask_better]
        k_for_cand = k_for_cand[mask_better]
        cand_paths = cand_paths[mask_better]
        f_vals = f_vals[mask_better]

        # Ограничиваем луч top-K по f
        m = candidates.size(0)
        k_keep = min(beam_width, m)
        top_f, top_idx = torch.topk(f_vals, k_keep, largest=False)

        beam_states = candidates[top_idx]
        beam_g = new_g[top_idx]
        beam_paths = cand_paths[top_idx]

        if log:
            print(
                f"[gpu_beam] layer={layer:03d} beam_in={b} cand={m} kept={k_keep} best_len={best_len}",
                flush=True,
            )

        if best_len <= 2:
            break

    return best_path if best_path is not None else base_moves


@torch.no_grad()
def beam_improve_batch_h_gpu(
    perms: List[Iterable[int]],
    *,
    baseline_moves_fn: Callable[[Iterable[int]], List[int]],
    beam_width: int = 4096,
    depth: int = 4096,
    w: float = 0.5,
    device: Optional[str] = None,
    log: bool = False,
) -> List[List[int]]:
    """Batсhovый beam search на GPU для нескольких перестановок одной длины.

    perms: список перестановок одинаковой длины n.
    Возвращает список путей (list[int]) для каждой perm (улучшенный или baseline).
    """
    if not perms:
        return []

    start_lists = [list(p) for p in perms]
    n = len(start_lists[0])
    if any(len(p) != n for p in start_lists):
        raise ValueError("Все перестановки в batch должны иметь одинаковую длину n.")
    if n <= 1:
        return [[] for _ in start_lists]

    P = len(start_lists)

    # Базовые решения и верхние границы по длине для каждой perm
    base_moves_list: List[List[int]] = []
    best_len = torch.empty(P, dtype=torch.int64)
    for i, p in enumerate(start_lists):
        bm = baseline_moves_fn(p)
        base_moves_list.append(list(bm))
        best_len[i] = len(bm)

    dev = _as_device(device)
    best_len = best_len.to(dev)
    start = torch.tensor(start_lists, dtype=torch.long, device=dev)  # [P, n]
    target = torch.arange(n, dtype=torch.long, device=dev).unsqueeze(0)  # [1, n]

    # Beam: [P, B, n], g: [P, B], paths: [P, B, depth]
    B = beam_width
    beam_states = start.unsqueeze(1).expand(-1, B, -1).clone()  # [P, B, n]
    beam_g = torch.full((P, B), fill_value=depth + 1, dtype=torch.int64, device=dev)
    beam_g[:, 0] = 0  # только первый луч активен в начале
    beam_paths = torch.zeros((P, B, depth), dtype=torch.int16, device=dev)

    flip_idx = _get_flip_idx(n, dev)
    k_values = torch.arange(2, n + 1, dtype=torch.int64, device=dev)  # [n-1]
    num_k = k_values.numel()

    best_paths_tensor = torch.zeros((P, depth), dtype=torch.int16, device=dev)
    best_found = torch.zeros(P, dtype=torch.bool, device=dev)

    if log:
        print(
            f"[gpu_beam_batch] batch_size={P} n={n} bw={beam_width} depth={depth} w={w} device={dev}",
            flush=True,
        )
    # Логируем устройства ключевых тензоров для проверки гипотезы о device mismatch (H_batch_device)
    _debug_log(
        hypothesis_id="H_batch_device",
        location="src/heuristics/gpu_beam.py:beam_improve_batch_h_gpu:init",
        message="batch_beam_devices",
        data={
            "P": P,
            "n": n,
            "beam_width": beam_width,
            "depth": depth,
            "device": str(dev),
            "best_len_device": str(best_len.device),
        },
    )

    for layer in range(1, depth + 1):
        # Активны те beam, для которых g + 1 < best_len[p]
        # (потенциально могут улучшить baseline для своей perm).
        candidate_possible = beam_g + 1 < best_len.unsqueeze(1)
        if not bool(candidate_possible.any()):
            break

        active_idx = candidate_possible.nonzero(as_tuple=False)  # [M, 2] (p, b)
        M = active_idx.size(0)
        if M == 0:
            break

        perm_idx_active = active_idx[:, 0]  # [M]
        beam_idx_active = active_idx[:, 1]  # [M]

        states_active = beam_states[perm_idx_active, beam_idx_active, :]  # [M, n]
        g_active = beam_g[perm_idx_active, beam_idx_active]  # [M]
        paths_active = beam_paths[perm_idx_active, beam_idx_active, :]  # [M, depth]

        # Порождаем кандидатов для всех k векторизованно.
        parent_idx_flat = torch.arange(M, device=dev).repeat_interleave(num_k)  # [M*num_k]
        perm_idx_parent = perm_idx_active[parent_idx_flat]  # [M*num_k]
        k_for_cand = k_values.repeat(M)  # [M*num_k]

        states_3d = states_active.unsqueeze(1).expand(-1, num_k, -1)  # [M, num_k, n]
        idx_3d = flip_idx.unsqueeze(0).expand(M, -1, -1)  # [M, num_k, n]
        cand_3d = torch.gather(states_3d, 2, idx_3d)  # [M, num_k, n]
        candidates = cand_3d.reshape(M * num_k, n)  # [M*num_k, n]

        new_g = g_active[parent_idx_flat] + 1  # [M*num_k]

        # Пути кандидатов
        parent_paths = paths_active[parent_idx_flat]  # [M*num_k, depth]
        cand_paths = parent_paths.clone()
        row_idx = torch.arange(cand_paths.size(0), device=dev)
        step_idx = torch.clamp(new_g - 1, min=0, max=depth - 1)
        cand_paths[row_idx, step_idx] = k_for_cand.to(cand_paths.dtype)

        # Проверка решений
        solved_mask = (candidates == target).all(dim=1)
        if bool(solved_mask.any()):
            solved_indices = solved_mask.nonzero(as_tuple=False).squeeze(1)
            for idx in solved_indices.tolist():
                p = int(perm_idx_parent[idx].item())
                g_val = int(new_g[idx].item())
                if g_val < int(best_len[p].item()):
                    best_len[p] = g_val
                    best_found[p] = True
                    best_paths_tensor[p, :g_val] = cand_paths[idx, :g_val]

        # Оставляем только нерешённые кандидаты
        keep_mask = ~solved_mask
        if not bool(keep_mask.any()):
            break
        candidates = candidates[keep_mask]
        new_g = new_g[keep_mask]
        perm_idx_parent = perm_idx_parent[keep_mask]
        cand_paths = cand_paths[keep_mask]

        # Эвристика и f-оценка
        h_vals = _gap_h_batch(candidates).to(torch.float32)
        f_vals = new_g.to(torch.float32) + w * h_vals

        # Отбрасываем кандидатов, которые уже не могут улучшить baseline для своей perm
        best_len_float = best_len.to(torch.float32)
        mask_better = f_vals < best_len_float[perm_idx_parent]
        if not bool(mask_better.any()):
            break
        candidates = candidates[mask_better]
        new_g = new_g[mask_better]
        perm_idx_parent = perm_idx_parent[mask_better]
        cand_paths = cand_paths[mask_better]
        f_vals = f_vals[mask_better]

        # Собираем новые beam по каждой perm отдельно (segmented top-k).
        # Инициализируем "неактивными" значениями.
        new_beam_states = torch.zeros_like(beam_states)
        new_beam_g = torch.full_like(beam_g, fill_value=depth + 1)
        new_beam_paths = torch.zeros_like(beam_paths)

        for p in range(P):
            mask_p = perm_idx_parent == p
            if not bool(mask_p.any()):
                continue
            cand_p_states = candidates[mask_p]
            cand_p_g = new_g[mask_p]
            cand_p_paths = cand_paths[mask_p]
            cand_p_f = f_vals[mask_p]

            m_p = cand_p_states.size(0)
            k_keep = min(beam_width, m_p)
            top_f, top_idx = torch.topk(cand_p_f, k_keep, largest=False)

            sel_states = cand_p_states[top_idx]  # [k_keep, n]
            sel_g = cand_p_g[top_idx]  # [k_keep]
            sel_paths = cand_p_paths[top_idx]  # [k_keep, depth]

            new_beam_states[p, :k_keep, :] = sel_states
            new_beam_g[p, :k_keep] = sel_g
            new_beam_paths[p, :k_keep, :] = sel_paths

        beam_states = new_beam_states
        beam_g = new_beam_g
        beam_paths = new_beam_paths

        if log:
            active_after = (beam_g + 1 < best_len.unsqueeze(1)).sum().item()
            print(
                f"[gpu_beam_batch] layer={layer:03d} active_beams={int(active_after)}",
                flush=True,
            )

        # Если для всех perm найдено решение длины <=2, то можно остановиться.
        if bool((best_len <= 2).all()):
            break

    # Сбор результатов: если улучшение найдено, берём best_paths_tensor; иначе baseline.
    out_moves: List[List[int]] = []
    for p in range(P):
        if bool(best_found[p].item()):
            L = int(best_len[p].item())
            path_tensor = best_paths_tensor[p, :L]
            out_moves.append([int(x) for x in path_tensor.tolist()])
        else:
            out_moves.append(base_moves_list[p])
    return out_moves


