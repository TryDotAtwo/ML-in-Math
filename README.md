# Pancake Problem

Задача сортировки перестановки префиксными разворотами (pancake sorting). Репозиторий объединяет код из двух источников (блокнот и pancake_91584_final_edit), рефакторинг по модулям, тесты и скрещивания алгоритмов.

## Документация

- **Навигация для агентов:** [docs/00_AGENT_NAVIGATION.md](docs/00_AGENT_NAVIGATION.md)
- **Обзор задачи и источников:** [docs/01_PROJECT_OVERVIEW.md](docs/01_PROJECT_OVERVIEW.md)
- **История изменений:** [docs/02_HISTORY_CHANGES.md](docs/02_HISTORY_CHANGES.md)
- **Гипотезы и скрещивания:** [docs/03_HYPOTHESES.md](docs/03_HYPOTHESES.md)
- **Результаты:** [docs/04_RESULTS.md](docs/04_RESULTS.md)
- **Описание тестов:** [docs/05_TESTS.md](docs/05_TESTS.md)
- **План рефакторинга:** [docs/06_REFACTORING_PLAN.md](docs/06_REFACTORING_PLAN.md)

## Установка и тесты

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Структура

- `src/core/` — перестановки, ходы, классический pancake sort.
- `src/heuristics/` — эвристики (gap_h, breakpoints2), beam search.
- `src/notebook_search/` — эвристический поиск из блокнота (v3_1).
- `src/submission/` — оценка сабмита, best_solution.
- `src/crossings.py` — скрещивания: блокнот+beam, baseline+beam, единый пайплайн.
- **`baseline/`** — примеры сабмишенов: `submission.csv`, `sample_submission.csv` (id, permutation, solution, ~2405 строк).

Оригинальные файлы: `pancake_91584_final_edit.py`, `копия_блокнота__pancake_problem_.py` (не удаляются, используются как референс).

## Папка baseline/ — примеры сабмишенов

В `baseline/` лежат готовые сабмишены для сравнения и проверки.

| Файл | Описание |
|------|----------|
| `baseline/submission.csv` | Сабмишен (id, permutation, solution). |
| `baseline/sample_submission.csv` | То же с колонкой permutation — удобно для check-steps без отдельного test. |

**Примеры команд с baseline:**

```bash
# Оценка сабмита из baseline против классического baseline-алгоритма (нужен test с id и permutation)
python main.py evaluate --test test.csv --submission baseline/submission.csv

# Проверка корректности решений (sample_submission уже содержит permutation)
python main.py check-steps --submission baseline/sample_submission.csv

# Сводка по n (score, prob_step, potential)
python main.py compare --best baseline/submission.csv

# Объединить: за основу — baseline, поверх применить свой сабмит
python main.py merge --base baseline/submission.csv --partials submission.csv --out final.csv
```

## Пайплайн для Kaggle

Все запуски решателей используют **перестановки из `baseline/sample_submission.csv`** (одна и та же тестовая выборка). Полученный файл **`submission.csv`** (колонки `id`, `solution`) можно отправлять в соревнование.

1. **Источник перестановок:** `baseline/sample_submission.csv` (по умолчанию для команды `solve`). При желании можно указать свой файл с колонками `id`, `permutation`: `--test путь/к/файлу.csv`.

2. **Запуск решателя** (на перестановках из baseline):
   ```bash
   python main.py
   ```
   или явно:
   ```bash
   python main.py solve --method baseline --out submission.csv
   python main.py solve --method beam --out submission_beam.csv
   python main.py solve --method notebook --solver v3_1 --out submission_nb.csv
   ```

3. **Автосабмит на Kaggle с твоего аккаунта:**
   - Установи API: `pip install kaggle`
   - Токен: зайди в [Kaggle → Account → API → Create New Token](https://www.kaggle.com/settings), скачай `kaggle.json`.
   - Положи `kaggle.json` в папку:
     - Windows: `C:\Users\<твой_логин>\.kaggle\kaggle.json`
     - Linux/macOS: `~/.kaggle/kaggle.json`
   - Узнай слаг соревнования из URL: `https://www.kaggle.com/c/<competition>`.
   - Один раз задай переменную (или передавай каждый раз):
     - Windows (PowerShell): `$env:KAGGLE_COMPETITION="competition-slug"`
     - Windows (cmd): `set KAGGLE_COMPETITION=competition-slug`
     - Linux/macOS: `export KAGGLE_COMPETITION=competition-slug`
   - Отправка (два варианта):
     - **Флаг после solve:** решить и сразу отправить, если установлен `--submit`:
       ```bash
       python main.py solve --submit --competition <competition-slug>
       # или задай KAGGLE_COMPETITION и тогда:
       python main.py solve --submit
       ```
     - **Отдельная команда:** сначала решить, потом отправить:
       ```bash
       python main.py
       python main.py submit --file submission.csv --competition <competition-slug>
       ```
   Результат появится на странице соревнования в разделе Submissions.
