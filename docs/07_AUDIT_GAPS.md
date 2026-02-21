# Аудит: пробелы и соответствие оригиналам

Дата: 2025-02-21. Сравнение `src/` с `pancake_91584_final_edit.py` и `копия_блокнота__pancake_problem_.py`.

---

## 1. Что уже есть в src/ и соответствует

| Компонент | Где в src | Примечание |
|-----------|-----------|------------|
| parse_permutation | core/permutation.py | Единый, с поддержкой None/пустой строки |
| moves_to_str, moves_len, apply_move_copy, apply_moves, is_solved | core/moves.py | + solution_to_moves (обратный парсинг) |
| pancake_sort_moves | core/baseline.py | Классический жадный |
| breakpoints2, gap_h, mix_h, make_h | heuristics/h_functions.py | — |
| beam_improve_or_baseline_h | heuristics/beam.py | — |
| pancake_sort_v3_1, notebook_baseline_v3_1 | notebook_search/solvers.py | Только один солвер |
| evaluate_submission_vs_baseline | submission/evaluate.py | — |
| best_solution | submission/best.py | + опциональный best_path, авто score из solution |
| solve_notebook_then_beam, solve_baseline_then_beam, solve_unified | crossings.py | Скрещивания 2–4 |

---

## 2. Блокнот (копия_блокнота__pancake_problem_.py)

| Функция / блок | Назначение | Статус |
|----------------|------------|--------|
| pancake_sort_path | Жадный сортировщик, возвращает list[str] "Rk" | **Есть** — `core/baseline.pancake_sort_path` |
| pancake_sort_input | Интерактивный ввод ходов | **Нет** (низкий приоритет) |
| prob_step | Эвристика «шагов до цели» для анализа | **Есть** — `notebook_search/utils.prob_step` |
| compare | Сводка по best_df по n | **Есть** — `submission/compare.compare` |
| process_row | Прогон одной строки через func(perm, treshold), сохранение в файл | **Есть** — `submission/process_row.process_row` |
| revers_perm | Обратная перестановка + decode_dict | **Есть** — `notebook_search/utils.revers_perm` |
| Солверы v3_1, v3_5, v4 | Эвристический поиск с порогом | **Есть** — `notebook_search/solvers`, реестр `SOLVER_REGISTRY` |
| Солверы v1, v2, v2_1, v2_2_np, v3, v3_3, v2_3, v3_2, v3_4, v3_5_np, v3_6, recursive_* | Остальные варианты из блокнота | **Нет** — при необходимости копировать из оригинала |
| print_search | Визуализация permute_search | **Нет** (низкий приоритет) |
| check_steps | Проверка корректности решений в DataFrame | **Есть** — `submission/check_steps.check_steps` |

---

## 3. pancake_91584_final_edit.py

| Функция / блок | Назначение | Статус |
|----------------|------------|--------|
| select_cases_per_n | Выбор k случаев на каждое n из df | **Есть** — `heuristics/experiments.select_cases_per_n` |
| run_grid | Грид по alpha, w, beam_width, depth; таблица метрик | **Есть** — `heuristics/experiments.run_grid` |
| full_eval_top_cfgs | Полный прогон top_cfgs по test_df, CSV, resume | **Есть** — `heuristics/experiments.full_eval_top_cfgs` |
| save_progress | Сохранение progress_map (id, solution) в CSV | **Есть** — `submission/merge.save_progress` |
| merge_submissions_with_partials | Объединение base + partial по длине решения | **Есть** — `submission/merge.merge_submissions_with_partials` |
| RunRow (dataclass) | Строка результата run_grid | Не перенесён (run_grid возвращает DataFrame) |
| **ML-блок** | Модели, обучение, beam_improve_with_ml, граф Кэли и т.д. | **Нет** — опционально вынести в `src/ml/` при необходимости |

---

## 4. Качество кода и согласованность модулей

- **Контракт ходов**: везде в src используется `list[int]`; строковый формат только на границах (moves_to_str / solution_to_moves). Ок.
- **Импорты**: heuristics/beam использует `..core.moves` — корректно при запуске из корня или с PYTHONPATH.
- **best_solution**: в оригинале при `best_df is None` загружался из BEST_SUBMISSION_PATH; в src при отсутствии best_path берётся копия submission_df и при необходимости достраивается score. Логика сравнения (лучше = меньше score) сохранена.
- **process_row в блокноте**: ожидает `func(perm, treshold)` с возвратом `(moves_or_pair, _, mlen, iter)` и `steps = moves[0]` (для v3_1: первый элемент — пара (moves_tuple, stat), тогда steps = moves_tuple). Нужна единая конвенция или адаптер для разных форматов возврата солверов.

---

## 5. Достаточность и правильность md

- **00_AGENT_NAVIGATION**: структура актуальна; после добавления солверов и экспериментов стоит добавить ссылку на реестр солверов и на 07_AUDIT_GAPS.
- **01_PROJECT_OVERVIEW**: описание задачи и источников верное; можно добавить одну строку про «полный функционал см. 07_AUDIT_GAPS».
- **05_TESTS**: описание тестов и запуска корректно; для новых модулей (run_grid, check_steps, merge) нужно добавить пункты в описание тестов.
- **06_REFACTORING_PLAN**: план выполнен частично; отсутствуют run_grid, full_eval, merge, все солверы блокнота, ML. После доработок обновить таблицу «Источники по модулям».

---

## 6. Рекомендуемые действия

1. **Ноутбук**: добавить в `notebook_search` все солверы (хотя бы как реестр name → callable) и общие утилиты: revers_perm, process_row, check_steps, prob_step; в core или submission — pancake_sort_path (или алиас к moves_to_str(pancake_sort_moves(...))).
2. **91584 (без ML)**: перенести в heuristics или в отдельный `experiments`: select_cases_per_n, run_grid, full_eval_top_cfgs; в submission — _save_progress, merge_submissions_with_partials (без display(), с moves_len из core).
3. **ML**: оформить как опциональный модуль `src/ml/` (models, beam_ml, train, eval) с зависимостями torch/cayleypy и пометкой в README.
4. **Документация**: обновить 00_AGENT_NAVIGATION и 06_REFACTORING_PLAN после добавлений; в 05_TESTS добавить тесты для check_steps, run_grid (малый грид), merge.

После этого функционал будет полным и удобным для анализа, оценки и экспериментов.
