# История изменений

Хронология и суть правок (обновляется по мере работы).

## 2025-02-21

### Инициализация репозитория и документации

- Добавлена папка `docs/` с файлами:
  - `00_AGENT_NAVIGATION.md` — навигация для агентов.
  - `01_PROJECT_OVERVIEW.md` — обзор задачи и источников кода.
  - `02_HISTORY_CHANGES.md` — этот файл.
  - `03_HYPOTHESES.md` — гипотезы и план скрещиваний.
  - `04_RESULTS.md` — результаты (пока заглушка).
  - `05_TESTS.md` — описание тестов.
  - `06_REFACTORING_PLAN.md` — план модулей и рефакторинга.

### Рефакторинг и тесты

- Создана структура `src/`: `core/` (permutation, moves, baseline), `heuristics/` (h_functions, beam), `notebook_search/` (solvers v3_1), `submission/` (evaluate, best), модуль `crossings.py` для скрещиваний 2–4.
- Добавлены unit-тесты: `tests/unit/test_core.py`, `test_heuristics.py`, `test_notebook_solvers.py`.
- Добавлены интеграционные тесты: `tests/integration/test_pipeline.py` (форматы, evaluate, best_solution, crossings).
- Добавлен `requirements.txt` (numpy, pandas, pytest).
- Скрещивания: `solve_notebook_then_beam`, `solve_baseline_then_beam`, `solve_unified` в `src/crossings.py`.

### Планируемые следующие шаги

- Вынос путей (Colab/Drive) в конфиг или env при необходимости.
- Прогон тестов в окружении пользователя: `python -m pytest tests/ -v`.
- Заполнение `04_RESULTS.md` после прогонов на реальных данных.

---

*Ниже добавлять новые записи по датам.*
