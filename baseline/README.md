### Папка `baseline/`

Здесь лежат **оригинальные исходники и материалы**, служащие референсами для рефакторинга в `src/` и для теоретического анализа.

---

### Структура

- `копия_блокнота__pancake_problem_.py` — копия оригинального Kaggle/Colab-блокнота:
  - семейство эвристических солверов `v1`, `v2`, `v3_*`, `v4`, `recursive_*`;
  - утилиты `process_row`, `best_solution`, `revers_perm`, `check_steps`, `compare`, `prob_step`;
  - боевой режим: `pancake_sort_v4` с `treshold=2.6` → скор ≈ 89980 (без beam).
  - Подробно: `docs/11_BASELINE_NOTEBOOK.md`.

- `pancake_91584_final_edit.py` — копия оригинального Colab-скрипта 91584:
  - baseline pancake sort, эвристики `gap_h`, `breakpoints2`, `mix_h`, `make_h`;
  - beam search `beam_improve_or_baseline_h` (baseline + beam → 91584);
  - эксперименты `run_grid`, `full_eval_top_cfgs`;
  - ML-блок (Pilgrim, EmbMLP, beam_improve_with_ml, CayleyGraph и др.).
  - Подробно: `docs/12_BASELINE_91584.md`.

- `bouzy-pancake-cgw2015.pdf` — статья Bruno Bouzy:
  - экспериментальное исследование pancake-проблемы;
  - Monte-Carlo Search (MCS) с доменно-зависимыми симуляциями (EffSort, AltSort, BREF, FDEffSort, FDAltSort, FG);
  - улучшение экспериментальной аппроксимации до ≈1.04.
  - Подробно: `docs/10_BOUZY_PAPER.md`.

- `18433-77-21949-1-2-20210717.pdf` — дополнительный PDF по теме (сжатый бинарный формат, пока без автоматического конспекта).
  - Подробно: `docs/13_BASELINE_PAPER_2.md` (заглушка, ожидает ручного конспекта человеком).

---

### Связь с `src/`

- Основные алгоритмы и утилиты из этих файлов перенесены в:
  - `src/core/`, `src/heuristics/`, `src/notebook_search/`, `src/submission/`, `src/crossings/`, `src/ml/`.
- Карта переноса и пробелов: `docs/07_AUDIT_GAPS.md`.

