#!/usr/bin/env python3
"""
Обёртка над beam gap 4096: single (1 перестановка) или batch (батчи с градуировкой по n).

Запуск:
  python run_beam_gap_4096.py --limit 100 --out submission_beam_gap_4096.csv --device cuda

Тестирование (скрипт, который находит оптимальный батч по n):
  python run_beam_gap_4096.py --quick
  Остальные аргументы (--test, --out-dir и т.д.) пробрасываются в бенчмарк.
"""

from __future__ import annotations

import sys


def _dispatch() -> None:
    sys.argv[0] = "run_beam_gap_4096.py"

    # Тестирование: бенчмарк поиска оптимального батча по n (с --quick по умолчанию)
    #    if "--quick" in sys.argv:
    #        from scripts.runners import run_beam_gap_benchmark as _bench
    #        _bench.main()
    #        return

    # Режим 1: одна перестановка за раз (активен)
    from scripts.runners import run_beam_gap_4096_single as _mod
    _mod.main()

    # Режим 2: батчами с градуировкой по n (раскомментировать и закомментировать блок выше для использования)
    # from scripts.runners import run_beam_gap_4096_batch as _mod
    # _mod.main()


if __name__ == "__main__":
    _dispatch()
