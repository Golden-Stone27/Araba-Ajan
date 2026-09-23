"""
Generalised Advantage Estimation. d_t = terminated_t ∨ truncated_t; the truncation bootstrap γ·V(final_obs) is
already folded into r_t during collection, so a done step never looks at V(s_{t+1}) (the next episode's obs).

    δ_t = r_t + γ (1 − d_t) V(s_{t+1}) − V(s_t),   V(s_T) = last_values
    Â_t = δ_t + γλ (1 − d_t) Â_{t+1},              R_t = Â_t + V(s_t)
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_gae(rewards: Tensor, values: Tensor, dones: Tensor, last_values: Tensor, gamma: float, lam: float
                ) -> tuple[Tensor, Tensor]:
    """Vectorised over N, backward loop over T. Inputs [T, N] (+ last_values [N]); returns (adv, returns) [T, N]."""
    T, N = rewards.shape
    assert values.shape == (T, N) and dones.shape == (T, N) and last_values.shape == (N,)
    adv = torch.empty_like(rewards)
    running = torch.zeros_like(last_values)
    next_v = last_values
    for t in range(T - 1, -1, -1):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nonterminal * next_v - values[t]
        running = delta + gamma * lam * nonterminal * running
        adv[t] = running
        next_v = values[t]
    return adv, adv + values


def gae_reference(rewards, values, dones, last_values, gamma: float, lam: float) -> tuple[list, list]:
    """Slow pure-Python double loop (tests only). Nested lists [T][N] of floats."""
    T, N = len(rewards), len(rewards[0])
    adv = [[0.0] * N for _ in range(T)]
    for i in range(N):
        a_next = 0.0
        for t in reversed(range(T)):
            v_next = float(last_values[i]) if t == T - 1 else float(values[t + 1][i])
            nd = 1.0 - float(dones[t][i])
            delta = float(rewards[t][i]) + gamma * nd * v_next - float(values[t][i])
            a_next = delta + gamma * lam * nd * a_next
            adv[t][i] = a_next
    ret = [[adv[t][i] + float(values[t][i]) for i in range(N)] for t in range(T)]
    return adv, ret
