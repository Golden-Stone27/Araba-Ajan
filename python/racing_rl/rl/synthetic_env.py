"""
TargetReachVecEnv: Unity-free stand-in with the UnityVecEnv contract (gymnasium VectorEnv, SAME_STEP autoreset,
infos["final_obs"] / ["_final_obs"]). obs[0:2] is a target ~ U(−1, 1)² resampled every step, obs[2:] is noise;
r = −‖clip(a, −1, 1) − target‖². Episodes are truncated after `episode_len` decisions (exercises the bootstrap).
"""

from __future__ import annotations

import gymnasium
import numpy as np
from gymnasium.vector.utils import batch_space


class TargetReachVecEnv(gymnasium.vector.VectorEnv):
    metadata = {"autoreset_mode": gymnasium.vector.AutoresetMode.SAME_STEP}

    def __init__(self, num_envs: int = 64, episode_len: int = 50, seed: int | None = None, obs_dim: int = 26,
                 act_dim: int = 2, noise: float = 1.0):
        assert obs_dim >= act_dim and 0.0 <= noise <= 1.0
        self.num_envs, self.episode_len, self.obs_dim, self.act_dim, self.noise = num_envs, episode_len, obs_dim, \
            act_dim, noise
        self.single_observation_space = gymnasium.spaces.Box(-1.0, 1.0, (obs_dim,), np.float32)
        self.single_action_space = gymnasium.spaces.Box(-1.0, 1.0, (act_dim,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self._rng = np.random.default_rng(seed)
        self._t = np.zeros(num_envs, np.int64)
        self._obs = self._sample(num_envs)
        self._actions: np.ndarray | None = None
        self.closed = False

    def _sample(self, n: int) -> np.ndarray:
        obs = self._rng.uniform(-1.0, 1.0, (n, self.obs_dim)).astype(np.float32)
        obs[:, self.act_dim:] *= self.noise
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t[:] = 0
        self._obs = self._sample(self.num_envs)
        return self._obs.copy(), {}

    def step_async(self, actions) -> None:
        self._actions = np.asarray(actions, np.float32).reshape(self.num_envs, self.act_dim)

    def step_wait(self):
        assert self._actions is not None, "step_wait without step_async"
        a = np.clip(self._actions, -1.0, 1.0)
        self._actions = None
        target = self._obs[:, :self.act_dim]
        reward = -np.sum((a - target) ** 2, axis=1).astype(np.float32)
        self._t += 1
        terminated = np.zeros(self.num_envs, bool)
        truncated = self._t >= self.episode_len
        nxt = self._sample(self.num_envs)
        final_obs = np.zeros_like(nxt)
        if truncated.any():
            k = int(truncated.sum())
            final_obs[truncated] = nxt[truncated]  # last obs of the finished episode
            nxt[truncated] = self._sample(k)  # first obs of the next one (SAME_STEP)
            self._t[truncated] = 0
        self._obs = nxt
        infos = {"final_obs": final_obs, "_final_obs": truncated.copy()}
        return nxt.copy(), reward, terminated, truncated, infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close_extras(self, **kwargs) -> None:
        pass
