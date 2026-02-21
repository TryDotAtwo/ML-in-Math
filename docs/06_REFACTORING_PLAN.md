# План рефакторинга: модули и папки

Цель: один раз описать целевую структуру, затем реализовать по шагам. После рефакторинга этот файл можно использовать как актуальную карту модулей.

---

## Целевая структура

```
src/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── permutation.py   # parse_permutation, представление перестановки
│   ├── moves.py         # apply_move_copy, apply_moves, moves_to_str, moves_len, is_solved
│   └── baseline.py      # pancake_sort_moves (жадный классический)
├── heuristics/
│   ├── __init__.py
│   ├── h_functions.py   # gap_h, breakpoints2, mix_h, make_h
│   └── beam.py          # beam_improve_or_baseline_h, run_grid (опционально)
├── notebook_search/
│   ├── __init__.py
│   ├── solvers.py       # pancake_sort_v3_1, v3_5, v4 и др. (выбранные для скрещивания)
│   └── utils.py         # revers_perm, mask_step и общие вспомогательные
├── ml/
│   ├── __init__.py
│   ├── models.py        # Pilgrim, SimpleMLP, EmbMLP, get_model
│   └── beam_ml.py       # beam_improve_with_ml, обёртки для инференса
└── submission/
    ├── __init__.py
    ├── evaluate.py      # evaluate_submission_vs_baseline
    ├── best.py          # best_solution (из блокнота)
    └── merge.py         # merge_submissions_with_partials, сохранение прогресса
```

---

## Источники по модулям

| Модуль | Откуда брать |
|--------|----------------|
| core/permutation.py | parse_permutation из 91584 или блокнота (унифицировать). |
| core/moves.py | apply_move_copy, apply_moves, moves_to_str, moves_len, is_solved из 91584. |
| core/baseline.py | pancake_sort_moves из 91584. В блокноте аналог — pancake_sort_path (возвращает list[str]); привести к list[int]. |
| heuristics/h_functions.py | breakpoints2, gap_h, mix_h, make_h из 91584. |
| heuristics/beam.py | beam_improve_or_baseline_h, run_grid из 91584. |
| notebook_search/solvers.py | pancake_sort_v3_1, v3_5, v4 (и при необходимости v3_3, v3_6) из блокнота. Сигнатура: (perm, treshold=...) -> (moves, ...); moves привести к list[int]. |
| notebook_search/utils.py | revers_perm, mask_step и т.п. из блокнота. |
| ml/models.py | Pilgrim, SimpleMLP, EmbMLP, get_model из 91584. |
| ml/beam_ml.py | beam_improve_with_ml и зависимости (apply_move_copy, is_solved, gap_h — импорт из core/heuristics). |
| submission/evaluate.py | evaluate_submission_vs_baseline из 91584. |
| submission/best.py | best_solution из блокнота. |
| submission/merge.py | save_progress, merge_submissions_with_partials из 91584. |
| submission/check_steps.py | check_steps из блокнота (проверка решений в DataFrame). |
| submission/process_row.py | process_row из блокнота (прогон строки через солвер). |
| submission/compare.py | compare(best_df) из блокнота (сводка по n). |
| heuristics/experiments.py | select_cases_per_n, run_grid, full_eval_top_cfgs из 91584. |
| notebook_search/utils.py | revers_perm, prob_step, steps_from_solver_result. |
| notebook_search/solvers.py | v3_1, v3_5, v4, SOLVER_REGISTRY, get_solver. |

---

## Общие правила

- Пути к данным (CSV, Drive) не хардкодить: конфиг или переменные окружения, с дефолтами для локального запуска.
- Colab-специфику (drive.mount, tqdm.notebook) изолировать в скриптах запуска или обёртках, не в ядре библиотеки.
- Единый формат решения в коде: `list[int]` ходов; строковый формат "R2.R5" только на границах (ввод/вывод CSV, сабмиты).
- Типизация: по возможности аннотации для аргументов и возвращаемых значений (list[int], Callable и т.д.).

---

## Порядок внедрения

1. Создать `src/` и подпапки с `__init__.py`.
2. Реализовать `core/` (permutation, moves, baseline) и подключить unit-тесты.
3. Перенести heuristics (h_functions, beam) и добавить test_heuristics, test_beam.
4. Вынести выбранные солверы блокнота в notebook_search и test_notebook_solvers.
5. Добавить submission (evaluate, best, merge) и интеграционные тесты.
6. При необходимости перенести ml/ и beam_ml; оставить опциональный импорт torch/cayleypy.

После каждого шага — прогон тестов и обновление истории в `02_HISTORY_CHANGES.md`.
