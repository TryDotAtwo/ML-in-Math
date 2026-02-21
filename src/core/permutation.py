# -*- coding: utf-8 -*-
"""Парсинг перестановки из строки (формат CSV: "0,1,2" или "1,0,2")."""

from __future__ import annotations

from typing import List


def parse_permutation(raw: str | None) -> List[int]:
    """Преобразует строку с запятыми в список целых — перестановку.
    Пустая строка или None дают [].
    """
    if raw is None:
        return []
    raw = str(raw).strip()
    if raw == "":
        return []
    return [int(tok) for tok in raw.split(",") if tok.strip() != ""]
