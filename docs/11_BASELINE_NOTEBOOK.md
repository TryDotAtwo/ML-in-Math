### Оригинальный блокнот: `baseline/копия_блокнота__pancake_problem_.py`

Этот файл — перенос оригинального Kaggle/Colab-блокнота по pancake sorting. Здесь кратко описано:

- **что делает исходный скрипт;**
- **какие части уже перенесены в `src/`;**
- **что осталось только в baseline и пока не используется.**

---

### 1. Назначение и окружение

- Язык: Python, стиль «блокнот».
- Окружение: Colab / Kaggle:
  - читаются CSV c `id`, `permutation`;
  - решения пишутся в submission CSV;
  - есть ad-hoc код для логирования прогресса и проверок.
- Цель исходника:
  - реализовать семейство эвристических солверов (v1, v2, v3\_*, v4, recursive\_*);
  - выбрать лучший (v4 с `treshold=2.6`) и получить скор **89980** без beam;
  - поддержать утилиты: `process_row`, `best_solution`, `revers_perm`, `check_steps`, `compare`, `prob_step`.

---

### 2. Основные компоненты исходника

По аудитам в `docs/07_AUDIT_GAPS.md`:

| Компонент                            | Назначение                                           |
|--------------------------------------|------------------------------------------------------|
| `pancake_sort_path`                  | жадный сортировщик, возвращает `list[str] "Rk"`      |
| `pancake_sort_input`                | интерактивный ввод ходов                             |
| `prob_step`                          | эвристика «сколько шагов до цели»                    |
| `revers_perm`                        | обратная перестановка + `decode_dict`                |
| `compare`                            | сводка по best\_df по `n`                            |
| `process_row`                        | прогон одной строки через `func(perm, treshold)`     |
| `check_steps`                        | проверка корректности решений в DataFrame            |
| Солверы `v3_1`, `v3_5`, `v4`         | эвристический поиск с порогом                        |
| Солверы `v1`, `v2`, `v2_1`, ...      | ранние / экспериментальные варианты                  |
| `recursive_*`                        | рекурсивные варианты поиска                          |
| `print_search`                       | визуализация `permute_search`                        |

Ключевой боевой режим блокнота:

- **`pancake_sort_v4` с `treshold=2.6`**:
  - даёт целевой скор ~89980;
  - используется через `process_row(perm, pancake_sort_v4, treshold=2.6)` по всем строкам теста;
  - затем, при необходимости, делается merge с `best_df`.

---

### 3. Что перенесено в `src/`

Перенос выполнен модульно, см. `docs/06_REFACTORING_PLAN.md` и `docs/07_AUDIT_GAPS.md`.

- **Базовая логика и форматы**
  - `pancake_sort_path` → `src/core/baseline.py` (`pancake_sort_path`, `pancake_sort_moves`).
  - представление ходов (`"Rk"` ↔ `int`) → `src/core/moves.py` (`moves_to_str`, `solution_to_moves`, `moves_len`, `apply_moves`, `is_solved`).

- **Солверы блокнота**
  - `pancake_sort_v3_1`, `pancake_sort_v3_5`, `pancake_sort_v4` → `src/notebook_search/solvers.py`:
    - возвращают `(moves_tuple, permute_search, arr_max_len, total_iter)`;
    - есть обёртки `notebook_baseline_v3_1`, `notebook_baseline_v4` → `List[int]` ходов;
    - реестр `SOLVER_REGISTRY` + `get_solver(name)` для main/run\_best\_score.

- **Утилиты и пайплайн**
  - `prob_step` → `src/notebook_search/utils.py` (`prob_step`).
  - `revers_perm` → `src/notebook_search/utils.py` (`revers_perm`).
  - `process_row` → `src/submission/process_row.py` (единый контракт `row + solver → dict`).
  - `check_steps` → `src/submission/check_steps.py`.
  - `compare` → `src/submission/compare.py`.
  - `best_solution` → `src/submission/best.py` (расширено: можно передавать `best_path`, авто-досчёт `score`).

- **Интеграция с beam и пайплайнами**
  - Solvers `v3_1`, `v4` доступны через:
    - `main.py solve --method notebook --solver v3_1|v4 ...`;
    - `src/crossings.py` (`solve_notebook_then_beam`, `solve_unified`) как baseline для beam.
  - `run_best_score.py`:
    - режим `--mode notebook` = только v4, `treshold=2.6`, без beam (повторяет исходный блокнот по скору 89980).

---

### 4. Что остаётся только в baseline

Компоненты, которые **ещё не перенесены** или не используются:

- Дополнительные солверы:
  - `v1`, `v2`, `v2_1`, `v2_2_np`, `v3`, `v3_3`, `v2_3`, `v3_2`, `v3_4`, `v3_5_np`, `v3_6`, `recursive_*`.
  - Потенциально полезны как **альтернативные baseline** или источники idей для новых эвристик.

- Интерактив / визуализация:
  - `pancake_sort_input` (ручной ввод);
  - `print_search` (печать структуры `permute_search`).

- Специфичные для блокнота фрагменты:
  - прямой код чтения/записи CSV, завязанный на пути Kaggle/Colab;
  - отладочные принты, экспериментальные куски.

Эти части можно рассматривать как **архив экспериментов**; при необходимости отдельные идеи можно перенести в `src/notebook_search` или `src/heuristics`.

---

### 5. Как использовать эту информацию агенту

- Когда нужен **точный контекст** по исходному блокноту:
  - читать этот файл + `docs/07_AUDIT_GAPS.md (секция 2)`;
  - при необходимости открывать `baseline/копия_блокнота__pancake_problem_.py` только точечно (по функциям).
- При разработке новых гипотез:
  - ссылаться на конкретные функции исходника и их аналоги в `src/`;
  - явно отмечать в `03_HYPOTHESES.md`, если гипотеза использует только часть функционала блокнота (например, только v4, только `prob_step`, и т.п.).

