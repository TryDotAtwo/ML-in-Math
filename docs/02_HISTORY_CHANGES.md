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

### Уточнение: как в оригиналах получены скоры 89980 и 91584

- Проверены оба исходника: **в них нет комбинации блокнот+beam**.
- **89980** (блокнот): сабмит строится только из **pancake_sort_v4** с **treshold=2.6** (`process_row(perm, pancake_sort_v4, treshold=2.6)`). Beam не используется.
- **91584** (pancake_91584): сабмит — **baseline + beam search** с gap-эвристикой (alpha=0, w=0.5, beam 128×128). Блокнот не используется.
- В проекте: `run_best_score.py` переведён на режимы `--mode notebook` (только v4, цель 89980) и `--mode beam` (только baseline+beam, цель 91584). Режимы crossing оставлены как экспериментальные.
- Обновлены docs: `01_PROJECT_OVERVIEW.md`, `03_HYPOTHESES.md`, `04_RESULTS.md`, README. В main.py `--treshold` допускает float (2.6 для v4).

---

## 2025-02-26

### RL (reinforcement learning): политика и run_rl.py

- Добавлен модуль **`src/ml/`**: окружение MDP (`env.py`), политика π(a|s) (`policy.py`), обучение BC и опционально PG (`train.py`), инференс и fallback на baseline (`inference.py`). Цель — минимальный суммарный скор.
- Добавлен скрипт **`run_rl.py`**: команды `train`, `solve`, `evaluate`, `submit`, `full`. Команда **`full --train --evaluate --submit`** выполняет полный цикл: обучение политик по n из теста → решение теста → оценка против baseline → отправка на Kaggle.
- В **main.py** добавлен метод **`rl`** в solve: `--method rl`, `--rl-models`, `--rl-device`.
- В **requirements.txt** добавлен `torch>=1.9`.
- Добавлены тесты **`tests/unit/test_ml.py`** (env, policy, fallback; требуют torch).
- Обновлены **README.md**, **docs/00_AGENT_NAVIGATION.md**, **docs/03_HYPOTHESES.md** (раздел RL), **docs/04_RESULTS.md** (секция RL), **docs/05_TESTS.md**, **docs/06_REFACTORING_PLAN.md**, **docs/07_AUDIT_GAPS.md**, **docs/01_PROJECT_OVERVIEW.md**.

### Анализ результатов и логирование экспериментов

- Добавлен **`src/submission/experiment_log.py`**: `log_experiment`, `log_evaluate` — запись в `runs/experiment_results.jsonl` (timestamp, script, method, score, n_rows, baseline_score, gain и т.д.).
- **run_rl.py** и **main.py** после solve и evaluate вызывают логирование; по логу можно анализировать скоры и сравнивать методы.
- Добавлен **`docs/08_ANALYSIS.md`** — инструкция для агента: как читать `runs/experiment_results.jsonl`, сопоставлять с гипотезами из `03_HYPOTHESES.md`, обновлять `04_RESULTS.md` и выносить вердикты (гипотеза подтверждена / провалена / требует проверки) с обоснованием по данным.
- В **00_AGENT_NAVIGATION.md** добавлены строки: лог экспериментов, анализ результатов и выводы по гипотезам.

### Документация по исходникам и исследованиям

- Добавлен **`docs/09_TOMAS_ALG.md`** — конспект топового решения Tom Rokicki (история, эвристики, hard-стеки, hash-based BFS).
- Добавлен **`docs/10_BOUZY_PAPER.md`** — конспект статьи Bruno Bouzy (MCS, доменно-зависимые симуляторы EffSort/AltSort/BREF/FD*, экспериментальные аппроксимации).
- Добавлены **`docs/11_BASELINE_NOTEBOOK.md`** и **`docs/12_BASELINE_91584.md`** — описания исходных скриптов блокнота и 91584, с картой переноса в `src/`.
- В папке `baseline/` добавлен `README.md` с кратким описанием содержимого и ссылками на соответствующие md.
- Добавлен **`docs/13_BASELINE_PAPER_2.md`** — заглушка под дополнительный PDF (`baseline/18433-77-21949-1-2-20210717.pdf`), пока без автоматического конспекта; есть инструкция для человека, как его дописать.

---

## 2026-02-27

### Обновление документации: статус модулей и новые гипотезы из статей

- **`docs/06_REFACTORING_PLAN.md`** полностью переработан:
  - Добавлена таблица «Текущий статус реализации» с явным ✅/⚠️/⬜ по каждому модулю.
  - Добавлен раздел «Что осталось сделать» (приоритизировано): ключевой пункт — v3_5 и v4 в `notebook_search/solvers.py` (без них скор 89980 не воспроизводим).
  - Обновлено дерево структуры и правила.

- **`docs/03_HYPOTHESES.md`** обновлён:
  - В сводную таблицу добавлены столбцы **«Обоснование»** и **«Источник»** для всех существующих гипотез.
  - Добавлены **7 новых гипотез**, напрямую следующих из исследовательских статей:
    - **H_LD** — lock detection (1-step lookahead on gap_h); источник: Valenzano & Yang 2017.
    - **H_LDD** — dual-state LDD/2LDD для более точной нижней оценки; источник: Valenzano & Yang 2017.
    - **H_hard_bench** — генерация hard-бенчмарков (Self-Inverse, Short Cycles, Bootstrapping) для честного тестирования; источник: Valenzano & Yang 2017 + Rokicki.
    - **H_singletons** — tie-breaking по числу singletons в beam; источник: Rokicki (приписывает «почти весь выигрыш» этой идее).
    - **H_incremental** — инкрементальное представление стека (двусвязный список + инкрем. hash); источник: Rokicki.
    - **H_two_ended** — двунаправленный поиск (forward + backward одновременно); источник: Rokicki.
    - **H_MCS** — Monte Carlo Search с rollout-политиками; источник: Bouzy 2015 (~1.04 аппрокс. против ~1.22 жадного).
  - Все гипотезы содержат разделы: описание, обоснование, связь с кодом, способ проверки.

### Новые эвристики: count_singletons, ld_h (H_singletons, H_LD)

- В **`src/heuristics/h_functions.py`** добавлены:
  - `count_singletons(state)` — число элементов с gap по обе стороны (Rokicki: «почти весь выигрыш»). O(N).
  - `ld_h(state)` — Lock Detection heuristic (Valenzano & Yang 2017): gap_h + 1 для locked-состояний. Admissible, O(N).
  - `make_h_ld()` — фабрика LD-эвристики.
  - `make_h_singleton_tiebreak(base_h)` — обёртка для tie-breaking по singletons в beam.
- Обновлён **`src/heuristics/__init__.py`** — экспорт новых функций.

### Оркестратор исследований: run_research.py

- Добавлен **`run_research.py`**: единый скрипт для пакетного прогона всех доступных методов с поддержкой resume через `state.json`.
  - Профили `--profile quick` и `--profile full`.
  - Методы: baseline, beam (64/128), beam+LD, beam+singletons, notebook v3_1/v3_5/v4, unified, RL (опционально).
  - Для каждого метода: solve → check_steps → evaluate vs baseline.
  - Сборка `merged_best.csv` (минимум по id из всех методов).
  - Артефакты: `runs/research/submissions/`, `metrics/`, `reports/`, `summary.csv`.
  - Запуск: `python run_research.py` или `python run_research.py --profile full --limit 300`.

### Обновление документации

- **`docs/06_REFACTORING_PLAN.md`**: v3_5 и v4 отмечены как ✅, solvers.py переведён из ⚠️ в ✅.
- **`docs/00_AGENT_NAVIGATION.md`**: убрана устаревшая пометка «v4 не перенесён», добавлены строки про `run_research.py` и новые эвристики.
- **`README.md`**: добавлен раздел «Авто-исследования с resume: run_research.py», обновлено описание `notebook_search`.

---

*Ниже добавлять новые записи по датам.*
