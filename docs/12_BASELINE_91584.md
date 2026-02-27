### Оригинальный скрипт: `baseline/pancake_91584_final_edit.py`

Этот файл — перенос оригинального Colab-скрипта, давшего скор **91584** на соревновании (baseline + beam search). Здесь кратко:

- **что делает исходный скрипт;**
- **какие компоненты перенесены в `src/`;**
- **что относится к ML-блоку и пока не портировано.**

---

### 1. Назначение и окружение

- Язык: Python, стиль «монолитного» Colab-скрипта.
- Окружение: Colab, с зависимостями (numpy, pandas, torch, cayleypy, tqdm и др.), путями на Google Drive.
- Цели исходника:
  - реализовать:
    - классический жадный pancake sort;
    - эвристики (gap\_h, breakpoints2, mix\_h, make\_h);
    - **beam search** с эвристикой (baseline + beam → 91584);
    - **ML-блок** (модели Pilgrim, SimpleMLP, EmbMLP, beam\_improve\_with\_ml) на графе Кэли;
  - запустить эксперименты (run\_grid, full\_eval\_top\_cfgs);
  - собирать/сливать сабмиты, вести прогресс.

---

### 2. Основные компоненты исходника

По аудитам в `docs/07_AUDIT_GAPS.md` (секция 3):

| Компонент                 | Назначение                                                      |
|---------------------------|-----------------------------------------------------------------|
| `pancake_sort_moves`      | жадный baseline (классический pancake sort)                    |
| `gap_h`, `breakpoints2`   | эвристики качества состояния                                   |
| `mix_h`, `make_h`         | комбинированная эвристика и фабрика h-функций                  |
| `beam_improve_or_baseline_h` | beam search: улучшает baseline по эвристике h             |
| `select_cases_per_n`      | выбор k кейсов на каждое `n` из DataFrame                      |
| `run_grid`                | грид по (alpha, w, beam\_width, depth); собирает метрики       |
| `full_eval_top_cfgs`      | полный прогон лучших конфигов по всему тесту                   |
| `save_progress`           | сохранение progress\_map (id → solution) в CSV                 |
| `merge_submissions_with_partials` | merge base + partial сабмиты по длине решения      |
| ML-блок (`Pilgrim`, `EmbMLP`, `SimpleMLP`, `beam_improve_with_ml`, `CayleyGraph` и т.п.) | обучение и использование ML-эвристики |

Ключевой боевой режим:

- **baseline + beam search** (`beam_improve_or_baseline_h` с gap\_h, alpha=0, w=0.5, beam\_width=128, depth=128):
  - baseline = жадный pancake sort;
  - beam по эвристике h (gap\_h / mix\_h);
  - даёт скор **91584** (без использования блокнота).

---

### 3. Что перенесено в `src/`

См. `docs/06_REFACTORING_PLAN.md` и `docs/07_AUDIT_GAPS.md`.

- **Ядро и эвристики**
  - `pancake_sort_moves` → `src/core/baseline.py`.
  - `gap_h`, `breakpoints2`, `mix_h`, `make_h` → `src/heuristics/h_functions.py`.
  - `beam_improve_or_baseline_h` → `src/heuristics/beam.py` (адаптирован к `src/core`).

- **Эксперименты по beam**
  - `select_cases_per_n`, `run_grid`, `full_eval_top_cfgs` → `src/heuristics/experiments.py`.

- **Работа с сабмитами**
  - `save_progress`, `merge_submissions_with_partials` → `src/submission/merge.py`.

- **Скрещивания и пайплайн**
  - baseline + beam (91584-пайплайн) → `src/crossings.py` (`solve_baseline_then_beam`).
  - `main.py solve --method beam` и `run_best_score.py --mode beam`:
    - используют baseline + beam (gap, 128×128);
    - воспроизводят скор 91584 без блокнота.

- **RL (новое, поверх 91584)**  
  - Отдельно в проект добавлен RL-пайплайн (`src/ml/`, `run_rl.py`), который использует другие идеи (BC/PG), но в 07\_AUDIT\_GAPS отмечен как часть «ML/91584» в более широком смысле.

---

### 4. Что осталось в baseline/ только в этом скрипте

Основной крупный блок, который **не перенесён** в `src/`:

- **ML-блок 91584:**
  - модели: `Pilgrim`, `SimpleMLP`, `EmbMLP`, `get_model(cfg)`;
  - генерация данных на графе Кэли (`PermutationGroups.pancake`, `CayleyGraph`);
  - обучение моделей (random walks на графе, loss, early stopping);
  - `beam_improve_with_ml`: beam search, в котором эвристика дополнена/заменена ML-предсказанием.
  - скрипты запуска экспериментов с ML-эвристикой.

Причины пока не переносить:

- сильная зависимость от внешних библиотек (torch, cayleypy);
- сложность воспроизводимости (данные, веса моделей, ресурсы GPU);
- приоритет — сначала стабилизировать не-ML пайплайны (baseline, notebook, beam, RL-политика).

Также остаются:

- Colab/Kaggle-специфичные части (пути к Google Drive, визуализация, ad-hoc код);
- старые отладочные фрагменты, не нужные в библиотечном `src/`.

---

### 5. Как это использовать агенту

- Для **понимания 91584-пайплайна**:
  - читать этот файл + `docs/07_AUDIT_GAPS.md (секция 3)` + `docs/06_REFACTORING_PLAN.md`;
  - исходник `baseline/pancake_91584_final_edit.py` открывать точечно.
- При разработке новых гипотез:
  - явно указывать, какие компоненты 91584 задействованы:
    - только baseline + beam (`heuristics/beam.py`);
    - или планируется перенос ML-блока в `src/ml/models.py`, `src/ml/beam_ml.py`.
- При расширении ML:
  - ссылаться на **ML-блок 91584** как на референс архитектур (EmbMLP, Pilgrim) и графовых данных;
  - переносить только то, что нужно под конкретную гипотезу (например, beam\_improve\_with\_ml для ограниченных `n`).

