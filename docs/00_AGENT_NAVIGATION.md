# Навигация для агентов (Pancake Problem)

Этот файл помогает другим агентам и разработчикам быстро ориентироваться в проекте.

## Структура репозитория

```
ML in Math/
├── docs/                    # Документация (читать в первую очередь)
│   ├── 00_AGENT_NAVIGATION.md   # этот файл
│   ├── 01_PROJECT_OVERVIEW.md   # что за задача, источники кода
│   ├── 02_HISTORY_CHANGES.md    # история изменений
│   ├── 03_HYPOTHESES.md         # гипотезы и цели (1–4)
│   ├── 04_RESULTS.md            # результаты экспериментов
│   ├── 05_TESTS.md              # описание тестов (unit, integration)
│   └── 06_REFACTORING_PLAN.md   # план модулей и папок
├── src/                     # Исходный код (после рефакторинга)
│   ├── core/                # ядро: перестановки, ходы, базовый сортировщик
│   ├── heuristics/          # эвристики и beam search (без ML)
│   ├── notebook_search/     # поиск из копии блокнота (v3_1, v3_5, v4, ...)
│   ├── ml/                  # ML-модели и beam_improve_with_ml
│   └── submission/          # сабмиты, оценка, merge
├── tests/                   # Тесты
│   ├── unit/
│   └── integration/
├── pancake_91584_final_edit.py   # оригинал (Colab/beam+ML)
└── копия_блокнота__pancake_problem_.py  # оригинал (блокнот, эвристический поиск)
```

## Где что искать

| Задача | Где смотреть |
|--------|---------------|
| Парсинг перестановки, применение хода, проверка решения | `src/core/` |
| Классический pancake sort (жадный O(n²)) | `src/core/baseline.py` или `pancake_sort_moves` в 91584 |
| Beam search с эвристикой (gap_h, breakpoints) | `src/heuristics/beam.py`, в 91584: `beam_improve_or_baseline_h` |
| Эвристический поиск с порогом (v3_1, v3_5, v4) | `src/notebook_search/`, в блокноте: `pancake_sort_v3_1` и др. |
| ML-модели (Pilgrim, EmbMLP) и beam+ML | `src/ml/`, в 91584: `beam_improve_with_ml`, `get_model` |
| Сравнение сабмитов, best_solution, merge | `src/submission/` (best, merge, evaluate, check_steps, process_row, compare) |
| Солверы блокнота (v3_1, v3_5, v4), реестр | `src/notebook_search/solvers.py`, `SOLVER_REGISTRY`, `get_solver` |
| Грид и полный прогон beam | `src/heuristics/experiments.py` (select_cases_per_n, run_grid, full_eval_top_cfgs) |
| Тесты корректности решений | `tests/unit/test_core.py`, `tests/integration/` |
| Пробелы и соответствие оригиналам | `docs/07_AUDIT_GAPS.md` |
| Примеры сабмишенов (baseline) | папка `baseline/` (submission.csv, sample_submission.csv) |

## Ключевые контракты

- **Перестановка**: `list[int]` длины n, значения 0..n-1 (identity = [0,1,...,n-1]).
- **Ход**: целое k от 2 до n включительно — разворот префикса длины k (Rk).
- **Решение**: `list[int]` ходов; строковый формат — `"R2.R5.R3"` (точка как разделитель).
- **Базовый солвер**: функция `(perm: Iterable[int]) -> list[int]` (ходы).

## Порядок чтения для нового агента

1. `docs/01_PROJECT_OVERVIEW.md` — контекст и источники.
2. `docs/06_REFACTORING_PLAN.md` — текущая разбивка по модулям.
3. `docs/05_TESTS.md` — как проверять корректность и что тестировать.
4. Код в `src/` по мере необходимости.

## Важные замечания

- Оригиналы `pancake_91584_final_edit.py` и `копия_блокнота__pancake_problem_.py` завязаны на Colab/Kaggle (пути, drive.mount). В `src/` пути и зависимости должны быть абстрагированы (конфиг, опциональный Google Drive).
- В блокноте `revers_perm(perm)` возвращает кортеж `(reversed_perm, decode_dict)`; при использовании в `process_row` с `from_target=True` нужен только первый элемент — учитывать при скрещивании.
