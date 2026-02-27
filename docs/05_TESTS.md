# Описание тестов

Тесты разнесены на **unit** и **integration**. Один файл = одна тема/модуль.

---

## Unit-тесты

Расположение: `tests/unit/`.

### test_core.py

- **parse_permutation**: пустая строка → [], "1,2,0" → [1,2,0], пробелы и запятые обрабатываются.
- **moves_to_str**: [2,5,3] → "R2.R5.R3"; [] → "".
- **moves_len**: строка "R2.R5" → 2; пустая/None → 0.
- **apply_move_copy**: один ход Rk даёт правильный разворот префикса; apply_moves(perm, []) = perm.
- **apply_moves**: последовательность ходов приводит к identity для известной перестановки.
- **is_solved**: [0,1,...,n-1] → True; любая другая перестановка → False.
- **pancake_sort_moves** (или аналог из core): для нескольких маленьких perm результат корректен (apply_moves(perm, sol) == list(range(n))), длина решения разумная (например ≤ 2n для жадного).

### test_heuristics.py

- **gap_h**: для identity возврат 0 или ожидаемое значение; для известной перестановки — консистентность.
- **breakpoints2**: граничные случаи (n=1, n=2) и один-два примера.
- **mix_h**, **make_h**: вызов не падает, тип float.

### test_beam.py

- **beam_improve_or_baseline_h**: на маленькой perm (n=5–8) с baseline = pancake_sort_moves возвращает корректное решение (проверка apply_moves); длина ≤ baseline.

### test_notebook_solvers.py

- Для одного-двух солверов блокнота (например v3_1 с treshold=3): на маленькой perm возвращается кортеж (moves, ...); apply_moves(perm, moves) == list(range(n)).

### test_ml.py (RL; требуется torch)

- **PancakeEnv**: step(k) даёт правильный next_state, reward=-1, done при identity.
- **PolicyNet**: forward (batch, n) → (batch, n-1) логиты.
- **solve_with_rl_or_baseline**: при отсутствии модели для n возвращается решение baseline; корректность apply_moves(perm, moves) == identity.
- **load_policy_for_n**: при отсутствии файла модели возвращается None.

Запуск: `python -m pytest tests/unit/test_ml.py -v` (при установленном torch).

### test_check_steps, test_process_row (integration)

- check_steps: DataFrame с id, permutation, solution — возвращает список id с неверным решением; для корректных решений — пустой список.
- process_row: строка + func (например pancake_sort_v3_1) → dict с id, solution, score, mlen, iter; применение solution к permutation даёт identity.

---

## Интеграционные тесты

Расположение: `tests/integration/`.

### test_pipeline_submission.py

- Загрузка тестового CSV (или фикстура с 2–3 строками), прогон через единый солвер (например baseline или beam), проверка что в выходном DataFrame есть id, solution, что solution валидный (check_steps-стиль проверка).
- best_solution: два сабмита с разными score — в результате лучший выбран; при равных score — без регрессии.

### test_cross_notebook_beam.py

- Один кейс: perm из теста, baseline = солвер блокнота (v3_1), вызов beam_improve_or_baseline_h; проверка корректности и что длина ≤ длины решения блокнота (или равная).

### test_formats.py

- Совместимость форматов: решение как list[int] конвертируется в строку "R2.R5" и обратно (если есть парсер строки в list[int] ходов); применение этой строки к perm даёт identity.

---

## Запуск

Из корня проекта (`ML in Math/`):

- Unit: `python -m pytest tests/unit/ -v`
- Integration: `python -m pytest tests/integration/ -v`
- Все: `python -m pytest tests/ -v`

Требования: `pip install -r requirements.txt` (numpy, pandas, pytest, torch). Тесты `test_ml.py` требуют torch; при отсутствии torch они пропускаются (importorskip).

---

## Ожидаемое покрытие

- Core и форматы — 100% по критичным путям.
- Beam и солверы блокнота — хотя бы один сценарий успешного решения на маленьком n.
- Интеграция — сценарии end-to-end без падений и с проверкой корректности решений.
