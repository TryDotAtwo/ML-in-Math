# -*- coding: utf-8 -*-
"""Политика π(a|s): перестановка -> логиты по действиям k=2..n."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """Вход: (batch, n) индексы перестановки. Выход: (batch, n-1) логиты для k=2..n."""

    def __init__(
        self,
        n: int,
        emb_dim: int = 32,
        hidden: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.n = n
        self.emb = nn.Embedding(n, emb_dim)
        self.pos_emb = nn.Embedding(n, emb_dim)
        self.register_buffer("_pos_idx", torch.arange(n, dtype=torch.long))
        inp = n * emb_dim
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(inp, hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LayerNorm(hidden))
            inp = hidden
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(inp, n - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n) long, значения 0..n-1. Возвращает (B, n-1) логиты."""
        B = x.size(0)
        x = self.emb(x.clamp(0, self.n - 1))
        pos = self._pos_idx.unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos)
        x = x.reshape(B, -1)
        x = self.mlp(x)
        return self.head(x)


def policy_forward(
    model: PolicyNet,
    state: List[int],
    device: torch.device,
    greedy: bool = True,
) -> int:
    """Один шаг: state -> действие k (2..n). greedy=True -> argmax, иначе sample."""
    n = len(state)
    x = torch.tensor([state], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)
    logits = logits[0]
    if greedy:
        k = 2 + int(logits.argmax().item())
    else:
        probs = F.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, 1).item()
        k = 2 + idx
    return min(max(k, 2), n)
