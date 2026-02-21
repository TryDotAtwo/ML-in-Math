# -*- coding: utf-8 -*-
"""Эвристический поиск из блокнота (порог, приоритеты). Возвращает list[int] ходов."""

from __future__ import annotations

from typing import List, Tuple, Any


def pancake_sort_v3_1(perm, treshold: int = 3) -> Tuple[Tuple[int, ...], Any, int, int]:
    """Эвристический поиск с порогом. Возвращает (moves_tuple, permute_search, arr_max_len, total_iter)."""
    arr = list(perm)
    n = len(arr)

    arr_max_len = 0
    total_iter = 0

    permute_search = {tuple(arr): ((), 0)}

    target = tuple(i for i in range(n))

    def check_and_write(idx, _div=0):
        _stat = stat + _div
        if (not moves or (moves and moves[-1] != idx)) and min_stat + treshold >= _stat:
            arr_ = tuple(list_arr[idx - 1 :: -1] + list_arr[idx:])
            if arr_ not in permute_search or (
                len(permute_search[arr_][0]) > len(moves) and permute_search[arr_][1] > _stat
            ):
                permute_search[arr_] = (moves + (idx,), _stat)

    while target not in permute_search:

        stage_permute_search = permute_search.copy()

        min_stat = min(i[1] for i in permute_search.values())

        for arr in stage_permute_search:

            total_iter += 1

            moves = permute_search[arr][0]
            stat = permute_search[arr][1]

            list_arr = list(arr)

            left_value = list_arr[0] - 1 if list_arr[0] else None
            left_idx = list_arr.index(left_value) if left_value is not None else None

            if left_idx and left_idx == 1:
                for i in range(2, n):
                    if list_arr[i - 1] - 1 != list_arr[i]:
                        check_and_write(i, 2)
                        break
            elif left_idx and list_arr[left_idx - 1] + 1 != left_value:
                check_and_write(left_idx, 1)
            elif left_idx:
                check_and_write(left_idx + 1, 3)

            rigth_value = list_arr[0] + 1 if list_arr[0] < n - 1 else None
            rigth_idx = list_arr.index(rigth_value) if rigth_value else n

            if rigth_idx == 1:
                for i in range(2, n):
                    if list_arr[i - 1] + 1 != list_arr[i]:
                        check_and_write(i, 2)
                        break
            elif not rigth_value or list_arr[rigth_idx - 1] - 1 != rigth_value:
                check_and_write(rigth_idx, 1)
            elif rigth_value:
                check_and_write(rigth_idx + 1, 2)

            arr_max_len = max(arr_max_len, len(permute_search))
            permute_search.pop(arr)

    return permute_search[target], permute_search, arr_max_len, total_iter


def notebook_baseline_v3_1(perm, treshold: int = 3) -> List[int]:
    """Обёртка для использования v3_1 как baseline_moves_fn: возвращает только list[int] ходов."""
    result, _, _, _ = pancake_sort_v3_1(perm, treshold=treshold)
    moves_tuple, _ = result
    return list(moves_tuple)


def _mask_step_v35(perm_list, n):
    perm_ = perm_list.copy()
    step = [] if perm_[n - 1] == n - 1 else [n]
    while len(perm_) > 2:
        m = perm_.pop()
        if not (perm_[-1] - 1 == m or perm_[-1] + 1 == m):
            step.append(len(perm_))
    return step


def pancake_sort_v3_5(perm, treshold: int = 3):
    """Эвристический поиск с mask_step (v3_5 из блокнота). Возвращает ((moves, stat), search, mlen, iter)."""
    arr = list(perm)
    n = len(arr)

    arr_max_len = 0
    total_iter = 0
    permute_search = {tuple(perm): ((), 0)}
    target = tuple(i for i in range(n))

    def check_and_write(idx, _div=0):
        _stat = stat + _div
        if (not moves or (moves and moves[-1] != idx)) and min_stat + treshold >= _stat:
            arr_ = tuple(list_arr[idx - 1 :: -1] + list_arr[idx:])
            if arr_ not in permute_search or (
                len(permute_search[arr_][0]) > len(moves) and permute_search[arr_][1] > _stat
            ):
                permute_search[arr_] = (moves + (idx,), _stat)

    while target not in permute_search:
        min_stat = min(i[1] for i in permute_search.values())

        for arr in list(permute_search):
            total_iter += 1
            moves = permute_search[arr][0]
            stat = permute_search[arr][1]
            list_arr = list(arr)

            left_value = list_arr[0] - 1 if list_arr[0] else None
            left_idx = list_arr.index(left_value) if left_value is not None else None
            rigth_value = list_arr[0] + 1 if list_arr[0] < n - 1 else None
            rigth_idx = list_arr.index(rigth_value) if rigth_value else n

            rigth_step = left_step = False
            if not rigth_value or (rigth_idx != 1 and list_arr[rigth_idx - 1] - 1 != rigth_value):
                check_and_write(rigth_idx, 10 / n)
                rigth_step = True
            elif rigth_idx != 1:
                check_and_write(rigth_idx, 10)
            if left_idx and left_idx != 1 and list_arr[left_idx - 1] + 1 != left_value:
                check_and_write(left_idx, 10 / n)
                left_step = True
            elif left_idx and left_idx != 1:
                check_and_write(left_idx, 10)
            if not rigth_step or not left_step:
                for i in _mask_step_v35(list_arr, n):
                    check_and_write(i, 10)

            arr_max_len = max(arr_max_len, len(permute_search))
            permute_search.pop(arr)

    return permute_search[target], permute_search, arr_max_len, total_iter


def pancake_sort_v4(perm, treshold: int = 10):
    """Эвристический поиск v4 (с left_way). Возвращает ((moves, stat, way), None, mlen, iter)."""
    n = len(perm)
    arr_max_len = 0
    total_iter = 0
    new_min_stat = 0
    permute_search = {tuple(perm): ((), 0, True)}
    target = tuple(i for i in range(n))

    def mask_step(perm_list):
        perm_ = list(perm_list)
        nn = len(perm_)
        step = [] if perm_[nn - 1] == nn - 1 else [nn]
        while len(perm_) > 2:
            m = perm_.pop()
            if not (perm_[-1] - 1 == m or perm_[-1] + 1 == m):
                step.append(len(perm_))
        return step

    def check_and_write(idx, _div=0, left_way=True):
        nonlocal new_min_stat
        _stat = stat + _div
        if (not moves or (moves and moves[-1] != idx)) and min_stat + treshold >= _stat:
            arr_ = tuple(list(arr)[idx - 1 :: -1] + list(arr)[idx:])
            old_stat = permute_search.get(arr_)
            if not old_stat:
                permute_search[arr_] = (moves + (idx,), _stat, left_way)
                new_min_stat = min(new_min_stat, _stat)

    while permute_search and target not in permute_search:
        min_stat = new_min_stat
        new_min_stat = min_stat + n

        for arr in list(permute_search):
            total_iter += 1
            moves, stat, way = permute_search[arr]
            right_step = left_step = False

            left_value = arr[0] - 1 if arr[0] else None
            left_idx = arr.index(left_value) if left_value is not None else None
            right_value = arr[0] + 1 if arr[0] < n - 1 else None
            right_idx = arr.index(right_value) if right_value else n

            if not right_value or (right_idx != 1 and arr[right_idx - 1] - 1 != right_value):
                check_and_write(right_idx, 0.05)
                right_step = True
            elif right_idx != 1:
                check_and_write(right_idx, 2)
            if left_idx and left_idx != 1 and arr[left_idx - 1] + 1 != left_value:
                check_and_write(left_idx, 0.25)
                left_step = True
            elif left_idx and left_idx != 1:
                check_and_write(left_idx, 2)
            if not right_step or not left_step:
                for i in mask_step(arr):
                    check_and_write(i, 2)

            arr_max_len = max(arr_max_len, len(permute_search))
            permute_search.pop(arr)

    return permute_search[target], None, arr_max_len, total_iter


# Реестр солверов для анализа и оценки: имя -> (callable, default_treshold)
SOLVER_REGISTRY = {
    "v3_1": (pancake_sort_v3_1, 3),
    "v3_5": (pancake_sort_v3_5, 3),
    "v4": (pancake_sort_v4, 10),
}


def get_solver(name: str):
    """Возвращает (func, default_treshold) для имени из SOLVER_REGISTRY."""
    if name not in SOLVER_REGISTRY:
        raise KeyError(f"Unknown solver: {name}. Available: {list(SOLVER_REGISTRY.keys())}")
    return SOLVER_REGISTRY[name]
