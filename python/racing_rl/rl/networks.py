"""
ActorCritic: separate actor (O → 256 → 256 → A, output μ) and critic (O → 256 → 256 → 1), no shared trunk.
log σ is a state-independent nn.Parameter(A), clamped to [log_std_min, log_std_max]. Observation normalisation
(RunningMeanStd, float64) lives on the module as `obs_rms` and is saved with every checkpoint.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch
from torch import Tensor, nn

from . import distributions as D
from .config import PPOConfig
from .normalization import RunningMeanStd

_ACT = {"tanh": nn.Tanh, "silu": nn.SiLU}


def _layer(in_dim: int, out_dim: int, gain: float) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, activation: str, out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [_layer(d, h, math.sqrt(2.0)), _ACT[activation]()]
        d = h
    layers.append(_layer(d, out_dim, out_gain))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 26, act_dim: int = 2, hidden_sizes: tuple[int, ...] = (256, 256),
                 activation: str = "tanh", action_transform: str = "clip", log_std_init: float = -0.5,
                 log_std_min: float = -5.0, log_std_max: float = 1.0, norm_obs: bool = True, obs_clip: float = 10.0):
        super().__init__()
        if activation not in _ACT:
            raise ValueError(f"unknown activation {activation!r}")
        self.obs_dim, self.act_dim = int(obs_dim), int(act_dim)
        self.action_transform = action_transform
        self.log_std_min, self.log_std_max = float(log_std_min), float(log_std_max)
        self.norm_obs, self.obs_clip = bool(norm_obs), float(obs_clip)
        # Orthogonal init runs a LAPACK QR whose result depends on the thread count (~1e-6); pin it to one thread so
        # the same seed gives bit-identical weights whatever torch.set_num_threads() the caller left behind.
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            self.actor = mlp(obs_dim, tuple(hidden_sizes), act_dim, activation, 0.01)
            self.critic = mlp(obs_dim, tuple(hidden_sizes), 1, activation, 1.0)
        finally:
            torch.set_num_threads(threads)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        self.obs_rms = RunningMeanStd(obs_dim)
        self.nonfinite_obs = 0

    @classmethod
    def from_config(cls, cfg: PPOConfig, obs_dim: int = 26, act_dim: int = 2) -> ActorCritic:
        return cls(obs_dim, act_dim, cfg.hidden_sizes, cfg.activation, cfg.action_transform, cfg.log_std_init,
                   cfg.log_std_min, cfg.log_std_max, cfg.norm_obs, cfg.obs_clip)

    # ------------------------------------------------------------------ pieces
    def clamped_log_std(self) -> Tensor:
        return torch.clamp(self.log_std, self.log_std_min, self.log_std_max)

    def normalize_obs(self, obs_raw: np.ndarray, update_rms: bool) -> Tensor:
        """Raw obs f32[N, O] → normalised f32 tensor [N, O]. update_rms=True only during rollout collection."""
        obs = np.asarray(obs_raw, np.float32)
        assert obs.ndim == 2 and obs.shape[1] == self.obs_dim, f"obs shape {obs.shape}, expected [N, {self.obs_dim}]"
        if not np.isfinite(obs).all():  # a single NaN would poison the running statistics for good
            self.nonfinite_obs += int((~np.isfinite(obs)).any(axis=1).sum())
            warnings.warn(f"non-finite observations replaced by 0 (total agents affected: {self.nonfinite_obs})")
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.norm_obs:
            return torch.from_numpy(obs.copy())
        if update_rms:
            self.obs_rms.update(obs)
        return torch.from_numpy(self.obs_rms.normalize(obs, self.obs_clip).astype(np.float32))

    def to_env(self, u: Tensor) -> np.ndarray:
        return D.transform_action(u, self.action_transform).numpy().astype(np.float32, copy=False)

    # ------------------------------------------------------------------ protocol (M4)
    @torch.no_grad()
    def act_full(self, obs_raw: np.ndarray, update_rms: bool, deterministic: bool = False
                 ) -> tuple[np.ndarray, Tensor, Tensor, Tensor, Tensor]:
        """(a_env f32[N, A], u [N, A], logp [N], v [N], obs_n [N, O]); obs_n is what the buffer must store."""
        obs_n = self.normalize_obs(obs_raw, update_rms)
        mu = self.actor(obs_n)
        log_std = self.clamped_log_std()
        u = mu if deterministic else D.sample(mu, log_std)
        logp = D.log_prob(u, mu, log_std)
        v = self.critic(obs_n).squeeze(-1)
        return self.to_env(u), u, logp, v, obs_n

    def act(self, obs_raw: np.ndarray, update_rms: bool) -> tuple[np.ndarray, Tensor, Tensor, Tensor]:
        """Stochastic action: (a_env, u, logp, v). The normalised obs of the call is kept in self.last_obs_n."""
        a_env, u, logp, v, obs_n = self.act_full(obs_raw, update_rms)
        self.last_obs_n = obs_n
        return a_env, u, logp, v

    @torch.no_grad()
    def act_deterministic(self, obs_raw: np.ndarray) -> np.ndarray:
        """a = transform(μ); never updates the running statistics."""
        return self.to_env(self.actor(self.normalize_obs(obs_raw, update_rms=False)))

    def evaluate(self, obs_n: Tensor, u: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(logp [B], H [B], v [B]) for already-normalised observations (with grad)."""
        assert obs_n.ndim == 2 and obs_n.shape[1] == self.obs_dim, f"obs_n shape {tuple(obs_n.shape)}"
        assert u.shape == (obs_n.shape[0], self.act_dim), f"u shape {tuple(u.shape)}"
        mu = self.actor(obs_n)
        log_std = self.clamped_log_std()
        logp = D.log_prob(u, mu, log_std)
        ent = D.entropy(log_std, logp.shape)
        v = self.critic(obs_n).squeeze(-1)
        return logp, ent, v

    @torch.no_grad()
    def value(self, obs_raw: np.ndarray) -> Tensor:
        """V(s) [N] for raw observations; normalises WITHOUT updating the running statistics."""
        return self.critic(self.normalize_obs(obs_raw, update_rms=False)).squeeze(-1)

    def sigma(self) -> np.ndarray:
        return torch.exp(self.clamped_log_std()).detach().numpy().astype(np.float64)
