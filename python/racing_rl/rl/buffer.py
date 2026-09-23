"""RolloutBuffer: preallocated f32 tensors [T, N, …], GAE, flattened + randperm-shuffled minibatches."""

from __future__ import annotations

from typing import Iterator, NamedTuple

import numpy as np
import torch
from torch import Tensor

from .gae import compute_gae


class Minibatch(NamedTuple):
    obs: Tensor  # [B, O] normalised with the collection-time statistics
    u: Tensor  # [B, A] unclipped Gaussian sample
    logp: Tensor  # [B]
    values: Tensor  # [B]
    adv: Tensor  # [B] (not normalised; PPOTrainer normalises per minibatch)
    returns: Tensor  # [B]


def _as_f32(x, shape: tuple[int, ...], name: str) -> Tensor:
    t = torch.from_numpy(np.asarray(x, np.float32)) if not isinstance(x, Tensor) else x.detach().to(torch.float32)
    assert tuple(t.shape) == shape, f"{name}: shape {tuple(t.shape)}, expected {shape}"
    return t


class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int = 26, act_dim: int = 2):
        self.T, self.N, self.obs_dim, self.act_dim = int(num_steps), int(num_envs), int(obs_dim), int(act_dim)
        T, N = self.T, self.N
        z = lambda *s: torch.zeros(s, dtype=torch.float32)  # noqa: E731
        self.obs = z(T, N, obs_dim)
        self.u = z(T, N, act_dim)
        self.logp = z(T, N)
        self.values = z(T, N)
        self.rewards = z(T, N)
        self.dones = z(T, N)
        self.last_values = z(N)
        self.adv = z(T, N)
        self.returns = z(T, N)
        self.pos = 0
        self.gae_ready = False

    @property
    def size(self) -> int:
        return self.T * self.N

    @property
    def full(self) -> bool:
        return self.pos == self.T

    def reset(self) -> None:
        self.pos = 0
        self.gae_ready = False

    def add(self, obs_n, u, logp, values, rewards, dones) -> None:
        assert self.pos < self.T, "buffer full; call reset()"
        N, t = self.N, self.pos
        self.obs[t] = _as_f32(obs_n, (N, self.obs_dim), "obs")
        self.u[t] = _as_f32(u, (N, self.act_dim), "u")
        self.logp[t] = _as_f32(logp, (N,), "logp")
        self.values[t] = _as_f32(values, (N,), "values")
        self.rewards[t] = _as_f32(rewards, (N,), "rewards")
        self.dones[t] = _as_f32(dones, (N,), "dones")
        self.pos += 1
        self.gae_ready = False

    def compute_gae(self, last_values, gamma: float, lam: float) -> None:
        """Advantages and returns (computed in float64, stored as f32)."""
        assert self.full, f"buffer holds {self.pos}/{self.T} steps"
        self.last_values.copy_(_as_f32(last_values, (self.N,), "last_values"))
        adv, ret = compute_gae(self.rewards.double(), self.values.double(), self.dones.double(),
                               self.last_values.double(), gamma, lam)
        self.adv.copy_(adv)
        self.returns.copy_(ret)
        self.gae_ready = True

    def flat(self) -> Minibatch:
        B = self.size
        return Minibatch(self.obs.reshape(B, self.obs_dim), self.u.reshape(B, self.act_dim), self.logp.reshape(B),
                         self.values.reshape(B), self.adv.reshape(B), self.returns.reshape(B))

    def minibatches(self, mb_size: int, generator: torch.Generator | None = None) -> Iterator[Minibatch]:
        """One epoch: [T·N, …] shuffled with randperm, chunks of mb_size (a trailing chunk < 2 is dropped)."""
        assert self.gae_ready, "compute_gae() first"
        data = self.flat()
        B = self.size
        mb = min(int(mb_size), B)
        perm = torch.randperm(B, generator=generator)
        for start in range(0, B, mb):
            idx = perm[start:start + mb]
            if idx.numel() < 2:
                break
            yield Minibatch(*(x[idx] for x in data))
