# -*- coding: utf-8 -*-
"""Общие фикстуры: добавление корня проекта в sys.path."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
