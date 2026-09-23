"""
Rollout collection over a SAME_STEP VectorEnv (UnityVecEnv, MultiUnityVecEnv, TargetReachVecEnv), numpy-vectorised:

    a_env, u, logp, v, obs_n = ac.act_full(obs, update_rms=True)
    step_async(a_env); next_obs, r, term, trunc, infos = step_wait()
    r[trunc] += γ · V(infos["final_obs"][trunc]) ;  buf.add(obs_n, u, logp, v, r, term | trunc)
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .buffer import RolloutBuffer
from .config import PPOConfig
from .networks import ActorCritic


class EpisodeStats:
    """Per-agent running (undiscounted, un-bootstrapped) episode return / length; finished ones are queued."""

    def __init__(self, num_envs: int):
        self.ret = np.zeros(num_envs, np.float64)
        self.len = np.zeros(num_envs, np.int64)
        self.returns: list[float] = []
        self.lengths: list[int] = []
        self.env_steps = 0
        self.env_time_s = 0.0

    def update(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        self.ret += rewards
        self.len += 1
        if dones.any():
            self.returns.extend(self.ret[dones].tolist())
            self.lengths.extend(self.len[dones].tolist())
            self.ret[dones] = 0.0
            self.len[dones] = 0

    def pop(self) -> tuple[list[float], list[int]]:
        r, n = self.returns, self.lengths
        self.returns, self.lengths = [], []
        return r, n


def _stack_final(final_obs, mask: np.ndarray) -> np.ndarray:
    sel = final_obs[mask]
    if sel.dtype == object:  # gymnasium Sync/AsyncVectorEnv keep per-env arrays in an object array
        sel = np.stack(list(sel))
    return np.asarray(sel, np.float32)


def collect_rollout(envs, ac: ActorCritic, buf: RolloutBuffer, obs: np.ndarray, cfg: PPOConfig,
                    stats: EpisodeStats | None = None) -> tuple[np.ndarray, torch.Tensor]:
    """Fill `buf` (T steps × N agents) starting from `obs`. Returns (next obs, last_values = V(next obs))."""
    torch.set_num_threads(cfg.torch_threads_collect)
    buf.reset()
    split = hasattr(envs, "step_async")  # gymnasium SyncVectorEnv (tests only) has just step()
    for _ in range(buf.T):
        a_env, u, logp, v, obs_n = ac.act_full(obs, update_rms=cfg.norm_obs)
        t0 = time.perf_counter()
        if split:
            envs.step_async(a_env)
            next_obs, rew, term, trunc, infos = envs.step_wait()
        else:
            next_obs, rew, term, trunc, infos = envs.step(a_env)
        if stats is not None:
            stats.env_time_s += time.perf_counter() - t0
        rew = np.asarray(rew, np.float32).copy()
        term = np.asarray(term, bool)
        trunc = np.asarray(trunc, bool)
        dones = term | trunc
        if stats is not None:
            stats.update(rew, dones)
            stats.env_steps += len(rew)
        if trunc.any():
            rew[trunc] += cfg.gamma * ac.value(_stack_final(infos["final_obs"], trunc)).numpy()
        buf.add(obs_n, u, logp, v, rew, dones)
        obs = np.asarray(next_obs, np.float32)
    return obs, ac.value(obs)
