# Навигация для агентов (Pancake Problem)

Этот файл помогает другим агентам и разработчикам быстро ориентироваться в проекте.

---

## Обязательный промпт для агента (читать перед началом работы)

> **Важно:** перед началом любой новой сессии работы с проектом **обязательно** прочитай этот раздел целиком и следуй пунктам ниже.

1. **Сначала прочти документацию:**
   - `docs/01_PROJECT_OVERVIEW.md` — что за задача и исходники.
   - `docs/03_HYPOTHESES.md` — список гипотез, их ID и текущие статусы (особенно таблицу в начале).
   - `docs/06_REFACTORING_PLAN.md` — что реализовано и что ещё осталось сделать (статус модулей).
   - `docs/07_AUDIT_GAPS.md` — карта того, что уже перенесено из baseline/91584, а что ещё только в оригиналах.
   - При работе с теорией/исходниками: `docs/09_TOMAS_ALG.md`, `docs/10_BOUZY_PAPER.md`, `docs/11_BASELINE_NOTEBOOK.md`, `docs/12_BASELINE_91584.md`, `docs/13_BASELINE_PAPER_2.md`.

2. **Правила работы с PDF в `baseline/`:**
   - При появлении нового PDF в `baseline/`:
     - попытайся прочитать его через `Read` (если текст виден — сделай новый `docs/XX_*.md` с кратким конспектом: название, авторы, основная задача, методы, результаты, связь с проектом; используй структуру как в `10_BOUZY_PAPER.md`);
     - если PDF в сжатом бинарном формате и текст не извлекается (как `18433-77-21949-1-2-20210717.pdf`), создай **заглушку** md (как `docs/13_BASELINE_PAPER_2.md`) с явным описанием проблемы и инструкцией для человека;
   - после создания нового md по PDF **обязательно**:
     - дополни `baseline/README.md` (краткое описание + ссылка на md);
     - при необходимости добавь ссылку в `docs/00_AGENT_NAVIGATION.md` (в таблицу ниже или в этот промпт);
     - зафиксируй добавление в `docs/02_HISTORY_CHANGES.md`.
   - **Из каждой статьи извлекай новые гипотезы** и добавляй их в `docs/03_HYPOTHESES.md` с указанием источника.

3. **Перед изменениями кода:**
   - Убедись, что понимаешь, к какой гипотезе относятся изменения (ID из `03_HYPOTHESES.md`).
   - Не трогай файлы в `baseline/`, кроме чтения — изменения делаем только в `src/` и `docs/`.

4. **При запуске экспериментов:**
   - Для solve/evaluate используй `main.py`, `run_best_score.py` или `run_rl.py` так, чтобы они писали логи в `runs/experiment_results.jsonl` (это уже встроено).
   - После каждого значимого прогона:
     - обнови `docs/04_RESULTS.md` (таблицы результатов);
     - обнови статус соответствующих гипотез в `docs/03_HYPOTHESES.md` (таблица в начале и блок конкретной гипотезы).

5. **При переносе кода из baseline:**
   - Сверяйся с `docs/07_AUDIT_GAPS.md` и соответствующим `docs/11_*.md` / `docs/12_*.md`.
   - После переноса обязательно допиши изменения в:
     - `docs/02_HISTORY_CHANGES.md` (новая дата и список шагов);
     - `docs/06_REFACTORING_PLAN.md` (обнови статус модуля на ✅).

6. **Перед завершением работы:**
   - Убедись, что:
     - `runs/experiment_results.jsonl` отражает последние прогоны (если что-то запускалось);
     - `docs/03_HYPOTHESES.md` и `docs/04_RESULTS.md` согласованы с логом;
     - `docs/06_REFACTORING_PLAN.md` отражает текущий статус модулей;
     - новые/изменённые модули описаны хотя бы кратко в соответствующем `docs/*.md`.

Следование этому промпту критично: проект опирается на актуальные md-файлы и лог экспериментов, чтобы агенты (и люди) могли продолжать работу без потери контекста.

## Структура репозитория

```
ML in Math/
├── docs/                    # Документация (читать в первую очередь)
│   ├── 00_AGENT_NAVIGATION.md   # этот файл
│   ├── 01_PROJECT_OVERVIEW.md   # что за задача, источники кода
│   ├── 02_HISTORY_CHANGES.md    # история изменений
│   ├── 03_HYPOTHESES.md         # гипотезы и цели (с источниками и обоснованием)
│   ├── 04_RESULTS.md            # результаты экспериментов
│   ├── 05_TESTS.md              # описание тестов (unit, integration)
│   ├── 06_REFACTORING_PLAN.md   # план модулей и папок (со статусом ✅/⚠️/⬜)
│   ├── 07_AUDIT_GAPS.md         # пробелы между src/ и оригиналами baseline
│   ├── 08_ANALYSIS.md           # как анализировать результаты и обновлять выводы по гипотезам
│   ├── 09_TOMAS_ALG.md          # конспект топового решения Tom Rokicki (singletons, hash-BFS, hard-стеки)
│   ├── 10_BOUZY_PAPER.md        # конспект Bouzy 2015 (MCS, rollout-симуляторы, ~1.04 аппрокс.)
│   ├── 11_BASELINE_NOTEBOOK.md  # описание baseline/копия_блокнота: компоненты и статус переноса
│   ├── 12_BASELINE_91584.md     # описание baseline/pancake_91584: компоненты и статус переноса
│   └── 13_BASELINE_PAPER_2.md   # конспект Valenzano & Yang 2017 (gap_h анализ, LD/2LD, hard-бенчмарки)
├── src/                     # Исходный код (после рефакторинга)
│   ├── core/                # ядро: перестановки, ходы, базовый сортировщик
│   ├── heuristics/          # эвристики и beam search (без ML)
│   ├── notebook_search/     # поиск из копии блокнота (v3_1, v3_5, v4)
│   ├── ml/                  # RL-политика и инференс (env, policy, train, inference)
│   └── submission/          # сабмиты, оценка, merge, логирование экспериментов
├── tests/                   # Тесты
│   ├── unit/
│   └── integration/
├── runs/                    # Результаты прогонов (experiment_results.jsonl, rl_models/, CSV)
├── baseline/                # Оригинальные скрипты и статьи (только чтение)
├── run_rl.py                # RL-пайплайн (train/solve/evaluate/submit/full)
├── run_research.py          # Пакетный прогон всех методов с resume (state.json)
├── main.py                  # Главная точка входа (solve/evaluate/submit)
└── run_best_score.py        # Воспроизведение оригинальных скоров (notebook/beam)
```

## Где что искать

| Задача | Где смотреть |
|--------|--------------|
| Парсинг перестановки, применение хода, проверка решения | `src/core/` |
| Классический pancake sort (жадный O(n²)) | `src/core/baseline.py` или `pancake_sort_moves` в 91584 |
| Beam search с эвристикой (gap_h, breakpoints) | `src/heuristics/beam.py`, в 91584: `beam_improve_or_baseline_h` |
| Эвристический поиск с порогом (v3_1, v3_5, v4) | `src/notebook_search/`, в блокноте: `pancake_sort_v3_1` и др. |
| ML-модели (Pilgrim, EmbMLP) и beam+ML | `src/ml/`, в 91584: `beam_improve_with_ml`, `get_model` |
| RL: политика π(a\|s), BC, инференс | `src/ml/` (env, policy, train, inference), `run_rl.py` (train/solve/evaluate/submit) |
| Сравнение сабмитов, best_solution, merge | `src/submission/` (best, merge, evaluate, check_steps, process_row, compare) |
| Солверы блокнота (v3_1, v3_5, v4), реестр | `src/notebook_search/solvers.py`, `SOLVER_REGISTRY`, `get_solver` |
| Грид и полный прогон beam | `src/heuristics/experiments.py` (select_cases_per_n, run_grid, full_eval_top_cfgs) |
| Тесты корректности решений | `tests/unit/test_core.py`, `tests/integration/` |
| Статус реализации модулей (что сделано / что осталось) | `docs/06_REFACTORING_PLAN.md` |
| Пробелы и соответствие оригиналам | `docs/07_AUDIT_GAPS.md` |
| Теория и топ-1 решение (Tom Rokicki) | `docs/09_TOMAS_ALG.md` (алгоритмы, singletons, hard-стеки) |
| MCS и rollout-стратегии (Bouzy 2015) | `docs/10_BOUZY_PAPER.md` |
| gap_h анализ, LD/LDD, hard-бенчмарки (Valenzano & Yang 2017) | `docs/13_BASELINE_PAPER_2.md` |
| Новые гипотезы из статей (H_LD, H_singletons, H_MCS и др.) | `docs/03_HYPOTHESES.md` → раздел «Новые гипотезы из исследовательских статей» |
| Примеры сабмишенов (baseline) | папка `baseline/` (submission.csv, sample_submission.csv) |
| Лог экспериментов (скоры, метод, gain) | `runs/experiment_results.jsonl` (пишется при solve/evaluate) |
| Анализ результатов и выводы по гипотезам | `docs/08_ANALYSIS.md` → обновлять `04_RESULTS.md` и итоги в `03_HYPOTHESES.md` |

## Ключевые контракты

- **Перестановка**: `list[int]` длины n, значения 0..n-1 (identity = [0,1,...,n-1]).
- **Ход**: целое k от 2 до n включительно — разворот префикса длины k (Rk).
- **Решение**: `list[int]` ходов; строковый формат — `"R2.R5.R3"` (точка как разделитель).
- **Базовый солвер**: функция `(perm: Iterable[int]) -> list[int]` (ходы).

## Порядок чтения для нового агента

1. `docs/01_PROJECT_OVERVIEW.md` — контекст и источники.
2. `docs/06_REFACTORING_PLAN.md` — текущий статус реализации (что готово, что нет).
3. `docs/03_HYPOTHESES.md` — все гипотезы с источниками и статусами.
4. `docs/05_TESTS.md` — как проверять корректность и что тестировать.
5. Код в `src/` по мере необходимости.

## Важные замечания

- Оригиналы `pancake_91584_final_edit.py` и `копия_блокнота__pancake_problem_.py` завязаны на Colab/Kaggle (пути, drive.mount). В `src/` пути и зависимости должны быть абстрагированы (конфиг, опциональный Google Drive).
- В блокноте `revers_perm(perm)` возвращает кортеж `(reversed_perm, decode_dict)`; при использовании в `process_row` с `from_target=True` нужен только первый элемент — учитывать при скрещивании.
- Солвер **v4** (и v3_5) перенесён в `src/notebook_search/solvers.py`. Скор 89980 воспроизводим через `run_best_score.py --mode notebook`.
- **`run_research.py`** — пакетный оркестратор: `python run_research.py` запускает все доступные методы, сравнивает с baseline, собирает merged-best и сохраняет `state.json` для resume.
- В `h_functions.py` добавлены `count_singletons` (H_singletons, Rokicki) и `ld_h` (H_LD, Valenzano & Yang 2017).
