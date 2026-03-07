# План рефакторинга: модули и папки

Цель: модульная структура `src/`, воспроизводящая весь функционал оригинальных файлов `baseline/`. Этот файл — живая карта состояния реализации.

---

## Текущий статус реализации

| Модуль | Файл(ы) | Статус | Примечание |
|--------|---------|--------|------------|
| core/permutation | `src/core/permutation.py` | ✅ Реализован | parse_permutation, поддержка None/пустой строки |
| core/moves | `src/core/moves.py` | ✅ Реализован | apply_move_copy, apply_moves, moves_to_str, moves_len, is_solved, solution_to_moves |
| core/baseline | `src/core/baseline.py` | ✅ Реализован | pancake_sort_moves, pancake_sort_path |
| heuristics/h_functions | `src/heuristics/h_functions.py` | ✅ Реализован | gap_h, breakpoints2, mix_h, make_h |
| heuristics/beam | `src/heuristics/beam.py` | ✅ Реализован | beam_improve_or_baseline_h |
| heuristics/experiments | `src/heuristics/experiments.py` | ✅ Реализован | select_cases_per_n, run_grid, full_eval_top_cfgs |
| notebook_search/solvers | `src/notebook_search/solvers.py` | ✅ Реализован | v3_1, v3_5, v4 + SOLVER_REGISTRY; v3_3, v3_6 — опционально |
| notebook_search/utils | `src/notebook_search/utils.py` | ✅ Реализован | revers_perm, prob_step, steps_from_solver_result |
| ml/env | `src/ml/env.py` | ✅ Реализован | PancakeEnv (MDP) |
| ml/policy | `src/ml/policy.py` | ✅ Реализован | PolicyNet π(a\|s), policy_forward |
| ml/train | `src/ml/train.py` | ✅ Реализован | train_bc, train_pg_epochs, train_bc_all_n |
| ml/inference | `src/ml/inference.py` | ✅ Реализован | load_policy_for_n, rollout_policy, solve_with_rl_or_baseline |
| ml/models | `src/ml/models.py` | ⬜ Не реализован | Pilgrim, EmbMLP, get_model из 91584 — опционально |
| ml/beam_ml | `src/ml/beam_ml.py` | ⬜ Не реализован | beam_improve_with_ml из 91584 — опционально |
| submission/evaluate | `src/submission/evaluate.py` | ✅ Реализован | evaluate_submission_vs_baseline |
| submission/best | `src/submission/best.py` | ✅ Реализован | best_solution, best_path |
| submission/merge | `src/submission/merge.py` | ✅ Реализован | save_progress, merge_submissions_with_partials |
| submission/check_steps | `src/submission/check_steps.py` | ✅ Реализован | check_steps |
| submission/process_row | `src/submission/process_row.py` | ✅ Реализован | process_row |
| submission/compare | `src/submission/compare.py` | ✅ Реализован | compare |
| submission/experiment_log | `src/submission/experiment_log.py` | ✅ Реализован | log_experiment, log_evaluate, analyze_results |
| crossings | `src/crossings.py` | ✅ Реализован | solve_notebook_then_beam, solve_baseline_then_beam, solve_unified |

**Итог:** 20 из 21 модулей реализованы (v3_5 и v4 добавлены в solvers.py). Не реализован только опциональный ML-блок из 91584 (`models.py`, `beam_ml.py`). В `h_functions.py` добавлены `count_singletons`, `ld_h`, `make_h_ld`, `make_h_singleton_tiebreak` (гипотезы H_singletons, H_LD).

---

## Что осталось сделать (приоритизировано)

| Приоритет | Задача | Где | Зависимость |
|-----------|--------|-----|-------------|
| ✅ Сделано | ~~Добавить солверы v3_5, v4~~ | `src/notebook_search/solvers.py` | Готово: v3_1, v3_5, v4 в SOLVER_REGISTRY |
| 🟡 Средний | Добавить солверы v3_3, v3_6 в реестр | `src/notebook_search/solvers.py` | Исходник: блокнот |
| 🟢 Низкий | Перенести ML-блок (Pilgrim, EmbMLP, get_model) | `src/ml/models.py` | torch, cayleypy; источник: 91584 |
| 🟢 Низкий | Перенести beam_improve_with_ml | `src/ml/beam_ml.py` | models.py; источник: 91584 |

**v3_5 и v4 перенесены.** Скор 89980 теперь воспроизводим: `python run_best.py --mode notebook` (v4, treshold=2.6).

---

## Целевая структура (справка)

```
src/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── permutation.py   ✅
│   ├── moves.py         ✅
│   └── baseline.py      ✅
├── heuristics/
│   ├── __init__.py
│   ├── h_functions.py   ✅
│   ├── beam.py          ✅
│   └── experiments.py   ✅
├── notebook_search/
│   ├── __init__.py
│   ├── solvers.py       ✅ (v3_1, v3_5, v4)
│   └── utils.py         ✅
├── ml/
│   ├── __init__.py
│   ├── env.py           ✅
│   ├── policy.py        ✅
│   ├── train.py         ✅
│   ├── inference.py     ✅
│   ├── models.py        ⬜ опционально
│   └── beam_ml.py       ⬜ опционально
├── submission/
│   ├── __init__.py
│   ├── evaluate.py      ✅
│   ├── best.py          ✅
│   ├── merge.py         ✅
│   ├── check_steps.py   ✅
│   ├── process_row.py   ✅
│   ├── compare.py       ✅
│   └── experiment_log.py ✅
├── crossings.py         ✅
└── __init__.py
```

---

## Общие правила

- Пути к данным (CSV, Drive) не хардкодить: конфиг или переменные окружения, с дефолтами для локального запуска.
- Colab-специфику (drive.mount, tqdm.notebook) изолировать в скриптах запуска или обёртках, не в ядре библиотеки.
- Единый формат решения в коде: `list[int]` ходов; строковый формат "R2.R5" только на границах (ввод/вывод CSV, сабмиты).
- Типизация: по возможности аннотации для аргументов и возвращаемых значений.

---

## Порядок внедрения (что осталось)

1. ~~Добавить v3_5 и v4~~ — **сделано**, в SOLVER_REGISTRY.
2. **[Опционально]** Добавить v3_3, v3_6 в реестр — расширяет опции для H_notebook_treshold и H_ensemble_multi_run.
3. **[Опционально, низкий приор]** Перенести ML-блок (models.py, beam_ml.py) при необходимости проверки H_ml (гипотеза о ML-beam из 91584).

После каждого шага — прогон тестов и обновление истории в `02_HISTORY_CHANGES.md`.
