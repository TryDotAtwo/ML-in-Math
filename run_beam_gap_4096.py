#!/usr/bin/env python3
"""
Тонкая обёртка над scripts.runners.run_beam_gap_4096.

Запускать так:
  python run_beam_gap_4096.py --limit 100 --out submission_beam_gap_4096.csv --device cuda

Флаг --device (cuda / cpu) и остальные параметры парсит scripts/runners/run_beam_gap_4096.py,
сюда аргументы просто проксируются без изменений.
"""

from __future__ import annotations

import sys


def _dispatch() -> None:
    from scripts.runners import run_beam_gap_4096 as _mod

    # Проксируем аргументы как есть, только имя скрипта меняем для help-сообщений.
    sys.argv[0] = "run_beam_gap_4096.py"
    _mod.main()


if __name__ == "__main__":
    _dispatch()

