"""Diagonal Gaussian with a state-independent log σ, and the u → env-action transforms (M4 math)."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

LOG_2PI = math.log(2.0 * math.pi)


def log_prob(u: Tensor, mu: Tensor, log_std: Tensor) -> Tensor:
    """log π(u|s) = Σ_j [ −(u_j−μ_j)²/(2σ_j²) − log σ_j − ½ log 2π ]  → shape [B]."""
    var = torch.exp(2.0 * log_std)
    return (-(u - mu) ** 2 / (2.0 * var) - log_std - 0.5 * LOG_2PI).sum(dim=-1)


def entropy(log_std: Tensor, batch_shape: torch.Size | tuple[int, ...]) -> Tensor:
    """H = Σ_j [ ½ + ½ log 2π + log σ_j ], broadcast to batch_shape (σ does not depend on the state)."""
    return (0.5 + 0.5 * LOG_2PI + log_std).sum(dim=-1).expand(batch_shape)


def sample(mu: Tensor, log_std: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """u = μ + σ · ε, ε ~ N(0, I)."""
    eps = torch.randn(mu.shape, generator=generator, dtype=mu.dtype, device=mu.device)
    return mu + torch.exp(log_std) * eps


def transform_action(u: Tensor, kind: str) -> Tensor:
    """clip: clip(u, −1, 1). mlagents_scale: clip(u, −3, 3) / 3 (ML-Agents continuous-action convention)."""
    if kind == "clip":
        return torch.clamp(u, -1.0, 1.0)
    if kind == "mlagents_scale":
        return torch.clamp(u, -3.0, 3.0) / 3.0
    raise ValueError(f"unknown action transform {kind!r}")


def transform_action_np(u: np.ndarray, kind: str) -> np.ndarray:
    if kind == "clip":
        return np.clip(u, -1.0, 1.0)
    if kind == "mlagents_scale":
        return np.clip(u, -3.0, 3.0) / 3.0
    raise ValueError(f"unknown action transform {kind!r}")
