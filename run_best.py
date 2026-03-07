#!/usr/bin/env python3
"""
Упрощённая точка входа для воспроизведения лучших скоров:
- notebook (v4, цель ~89980),
- beam (baseline+beam, цель ~91584),
- crossing-* (экспериментальные режимы блокнот+beam).

По сути это тонкая обёртка над scripts.runners.run_best_score.main.
"""

from scripts.runners.run_best_score import main


if __name__ == "__main__":
    main()

