"""RunningMeanStd: float64 running moments with the parallel (Chan et al.) merge of Welford's algorithm."""

from __future__ import annotations

import numpy as np
import torch

NORM_EPS = 1e-8


class RunningMeanStd:
    """
    Population mean / variance of everything seen so far. Starts empty (count 0), so the first update adopts the
    batch moments exactly: chunked updates equal np.mean / np.var of the concatenated data.
    """

    def __init__(self, shape: tuple[int, ...] | int):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.mean = np.zeros(self.shape, np.float64)
        self.var = np.ones(self.shape, np.float64)
        self.count = 0.0

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, np.float64).reshape(-1, *self.shape)
        if x.shape[0] == 0:
            return
        self.update_from_moments(x.mean(axis=0), x.var(axis=0), x.shape[0])

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
        if batch_count <= 0:
            return
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        m2 = self.var * self.count + batch_var * batch_count + delta * delta * (self.count * batch_count / tot)
        self.mean = self.mean + delta * (batch_count / tot)
        self.var = m2 / tot
        self.count = float(tot)

    def merge(self, other: RunningMeanStd) -> None:
        """In-place parallel merge (e.g. statistics gathered by separate workers)."""
        self.update_from_moments(other.mean, other.var, other.count)

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """clip((x − mean) / sqrt(var + 1e−8), −clip, clip), float64."""
        z = (np.asarray(x, np.float64) - self.mean) / np.sqrt(self.var + NORM_EPS)
        return np.clip(z, -clip, clip)

    def copy(self) -> RunningMeanStd:
        c = RunningMeanStd(self.shape)
        c.mean, c.var, c.count = self.mean.copy(), self.var.copy(), self.count
        return c

    def state_dict(self) -> dict:
        return {"mean": torch.from_numpy(self.mean.copy()), "var": torch.from_numpy(self.var.copy()),
                "count": float(self.count)}

    def load_state_dict(self, state: dict) -> None:
        mean = np.asarray(torch.as_tensor(state["mean"]).cpu().numpy(), np.float64).reshape(self.shape)
        var = np.asarray(torch.as_tensor(state["var"]).cpu().numpy(), np.float64).reshape(self.shape)
        self.mean, self.var, self.count = mean.copy(), var.copy(), float(state["count"])
