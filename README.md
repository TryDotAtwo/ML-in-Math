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
- `src/heuristics/` — эвристики (gap_h, breakpoints2, `ld_h`, singletons), beam search.
- `src/notebook_search/` — эвристический поиск из блокнота (v3_1, v3_5, v4).
- `src/submission/` — оценка сабмита, best_solution.
- `src/crossings.py` — скрещивания: блокнот+beam, baseline+beam, единый пайплайн.
- `src/ml/` — RL: окружение, политика π(a|s), обучение (BC + опционально PG), инференс.
- **`baseline/`** — примеры сабмишенов: `submission.csv`, `sample_submission.csv` (id, permutation, solution, ~2405 строк).

Оригинальные файлы: `pancake_91584_final_edit.py`, `копия_блокнота__pancake_problem_.py` (не удаляются, используются как референс).

## RL (минимальный скор): run_rl.py

Отдельный скрипт для обучения политики и сабмита с целью **минимального суммарного скора**.

Зависимость: `pip install torch` (уже в requirements.txt).

**Один запуск — полный цикл (обучение → решение → оценка → сабмит):**

```bash
python run_rl.py full --train --test baseline/sample_submission.csv --models runs/rl_models --out submission.csv --evaluate --submit
```

Команда `full --train` сначала обучает политики (BC, опционально PG) по всем n из теста, затем решает тест, при `--evaluate` сравнивает с baseline, при `--submit` отправляет файл на Kaggle. Без `--train` используются уже сохранённые модели.

**Отдельные команды:**

```bash
# Только обучить политики (BC; опционально --pg-epochs для дообучения)
python run_rl.py train --test baseline/sample_submission.csv --out-dir runs/rl_models

# Только решить тест (RL где есть модель, иначе baseline)
python run_rl.py solve --test baseline/sample_submission.csv --models runs/rl_models --out submission.csv

# Оценить submission против baseline
python run_rl.py evaluate --test baseline/sample_submission.csv --submission submission.csv

# Отправить на Kaggle
python run_rl.py submit --file submission.csv
```

Через main.py: `python main.py solve --method rl --rl-models runs/rl_models --out submission.csv`

## Авто-исследования с resume: run_research.py

Скрипт для пакетного прогона доступных методов, сравнения с baseline, проверки корректности решений и сборки `merged_best` с сохранением состояния (`state.json`) и продолжением с места остановки.

```bash
# Быстрый профиль (несколько методов), можно безопасно прерывать и запускать снова
python run_research.py --profile quick --out-dir runs/research

# Полный профиль с более тяжёлыми конфигурациями
python run_research.py --profile full --out-dir runs/research_full

# Ограничить размер теста для чернового прогона
python run_research.py --profile quick --limit 300
```

Артефакты:
- `runs/research/state.json` — состояние шагов (resume).
- `runs/research/submissions/*.csv` — сабмиты по методам.
- `runs/research/metrics/*.json` — метрики compare vs baseline + check-steps.
- `runs/research/summary.csv` и `runs/research/reports/latest_report.md` — сводка и мини-отчёт.

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

   **Скор как в оригиналах (в исходниках нет «блокнот+beam», каждый скор — одним методом):**
   - **89980** — блокнот: только **v4**, treshold=2.6, без beam.
   - **91584** — 91584: только **baseline + beam** (gap, 128×128), без блокнота.
   Команды:
   ```bash
   python run_best_score.py --mode notebook --out submission.csv   # цель 89980 (только v4)
   python run_best_score.py --mode beam --out submission.csv       # цель 91584 (только beam)
   ```
   Явно через main.py:
   ```bash
   python main.py solve --method notebook --solver v4 --treshold 2.6 --out submission.csv
   python main.py solve --method beam --out submission.csv
   ```
   Режимы `--mode crossing-memory` и `--mode crossing-quality` — экспериментальные (блокнот+beam). Оценка после решения:
   ```bash
   python main.py evaluate --test baseline/sample_submission.csv --submission submission.csv
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
