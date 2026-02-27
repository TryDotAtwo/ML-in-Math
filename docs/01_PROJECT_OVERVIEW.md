# Обзор проекта: Pancake Sorting

## Задача

**Pancake sorting** — задача сортировки перестановки префиксными разворотами. Дана перестановка `perm` длины n (числа 0..n-1). Разрешённый ход: выбрать k от 2 до n и развернуть первые k элементов (обозначение Rk). Цель — привести перестановку к тождественной [0,1,...,n-1] за минимальное число ходов.

Формат вывода: строка ходов, например `"R2.R5.R3"` (разделитель — точка).

## Как в оригиналах получают целевые скоры

**В обоих исходниках нет комбинации «блокнот + beam».** Каждый даёт хороший скор одним методом:

| Скор | Источник | Как получен |
|------|----------|-------------|
| **89980** | Блокнот (`копия_блокнота__pancake_problem_.py`) | Только **pancake_sort_v4**, treshold=**2.6**, без beam. Сабмит: `process_row(perm, pancake_sort_v4, treshold=2.6)` по всем строкам, затем при необходимости merge с best_df. |
| **91584** | pancake_91584 (`pancake_91584_final_edit.py`) | Только **baseline + beam search** с gap-эвристикой (alpha=0, w=0.5, beam 128×128). Блокнот не используется. Итоговый сабмит — merge baseline с submission_gap.csv (beam по всем 2405). |

Поэтому в проекте для воспроизведения оригиналов используются два отдельных режима: **notebook** (только v4) и **beam** (только baseline+beam). Режимы crossing-notebook-beam — экспериментальные (скрещивание двух подходов). Дополнительно реализован **RL-пайплайн** (политика π(a|s), обучение BC+PG, инференс с fallback на baseline): скрипт `run_rl.py`, команда `full --train --evaluate --submit` для полного цикла; см. README и `docs/04_RESULTS.md` (секция RL).

## Источники кода

1. **pancake_91584_final_edit.py** (Colab)
   - Классический жадный pancake sort.
   - Эвристики: `gap_h`, `breakpoints2`, `mix_h`.
   - Beam search с эвристикой: `beam_improve_or_baseline_h`, грид по alpha/w/beam_width/depth. **Итог 91584 — только baseline+beam, без блокнота.**
   - ML: модели Pilgrim, SimpleMLP, EmbMLP; обучение на графе Кэли; `beam_improve_with_ml`.
   - Сабмиты: прогресс в CSV, merge частичных результатов, оценка против baseline.

2. **копия_блокнота__pancake_problem_.py** (Colab/Kaggle)
   - Базовый pancake sort с выводом в виде строк "Rk".
   - Много вариантов эвристического поиска: v1, v2, v2_1, v3_1 … v4, recursive и др.
   - **Итог 89980 — только pancake_sort_v4 с treshold=2.6, без beam.**
   - Утилиты: `process_row`, `best_solution`, `revers_perm`, `check_steps`, `compare`, `prob_step`.

## Цели проекта (из постановки)

1. **Улучшить** — рефакторинг, тесты, документация, стабильность.
2. **Скрестить копию блокнота с beam search** — использовать beam search из 91584 как улучшатель решений, полученных солверами из блокнота (или наоборот).
3. **Скрестить копию блокнота с pancake_91584** — объединить утилиты, пайплайны, форматы; один конвейер из двух источников.
4. **Скрестить блокнот + beam search + pancake_91584** — полная интеграция: эвристический поиск блокнота + beam по эвристике + (опционально) ML-beam из 91584; общие форматы и тесты.

Полный перечень перенесённого функционала и оставшихся пробелов — в [docs/07_AUDIT_GAPS.md](07_AUDIT_GAPS.md). Теоретические детали:

- **[docs/09_TOMAS_ALG.md](09_TOMAS_ALG.md)** — конспект победившего подхода Tom Rokicki: hash-based BFS, singletons, hard-стеки (s5, l9, арифметические прогрессии), двунаправленный поиск.
- **[docs/10_BOUZY_PAPER.md](10_BOUZY_PAPER.md)** — конспект статьи Bouzy 2015: MCS с доменно-зависимыми rollout-симуляторами (~1.04 аппроксимация).
- **[docs/13_BASELINE_PAPER_2.md](13_BASELINE_PAPER_2.md)** — конспект Valenzano & Yang 2017: анализ gap_h, lock detection (LD/2LD/LDD), генерация hard-бенчмарков.

Гипотезы, следующие из этих работ: H_LD, H_LDD, H_hard_bench, H_singletons, H_incremental, H_two_ended, H_MCS — подробно в [docs/03_HYPOTHESES.md](03_HYPOTHESES.md).

## Термины

- **Baseline** — классический жадный pancake sort (O(n²) ходов).
- **Beam search** — поиск с ограничением по ширине луча и эвристикой f = g + w*h (g — длина пути, h — эвристика состояния).
- **Notebook search** — семейство солверов из блокнота (v3_1, v3_5, v4 и т.д.) с порогом и приоритетами.
- **ML-beam** — beam search, в котором эвристика дополнена или заменена выходом нейросети.
- **RL (run_rl.py)** — политика π(a|s), обучение behavior cloning и опционально policy gradient; инференс — один rollout до решения (низкая память). Полный цикл: `run_rl.py full --train --evaluate --submit`.
- **MCS (Monte Carlo Search)** — nested MCS с rollout-политиками; не реализован, является гипотезой H_MCS. Источник: Bouzy 2015.
- **LD/LDD** — lock detection: lookahead-улучшения gap_h, вычислимые за O(N). Не реализованы, гипотезы H_LD, H_LDD. Источник: Valenzano & Yang 2017.
- **Singletons** — блинчики с разрывами с обеих сторон; tie-breaking в beam по числу singletons. Гипотеза H_singletons. Источник: Rokicki.
