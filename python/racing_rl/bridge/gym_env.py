"""RacingEnv: single-agent gymnasium.Env over UnityVecEnv(num_agents=1)."""

from __future__ import annotations

import math
import os

import gymnasium
import numpy as np

from .vec_env import UnityVecEnv


class RacingEnv(gymnasium.Env):
    """
    One car. step() returns the finished episode's final observation when terminated/truncated (standard
    gymnasium.Env semantics); the next reset() always sends a RESET, seeded from self.np_random when seed=None.
    """

    metadata = {"render_modes": []}

    def __init__(self, exe_path: str | os.PathLike | None, num_agents: int = 1, **kwargs):
        if num_agents != 1:
            raise ValueError("RacingEnv wraps exactly one agent; use UnityVecEnv for N > 1")
        self.vec = UnityVecEnv(exe_path, num_agents=1, **kwargs)
        self.observation_space = self.vec.single_observation_space
        self.action_space = self.vec.single_action_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        s = seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        obs, infos = self.vec.reset(seed=s, options=options)
        return obs[0], _single(infos)

    def step(self, action):
        obs, rew, term, trunc, infos = self.vec.step(np.asarray(action, np.float32).reshape(1, -1))
        done = bool(term[0] or trunc[0])
        o = infos["final_obs"][0] if done else obs[0]
        return o, float(rew[0]), bool(term[0]), bool(trunc[0]), _single(infos)

    def close(self):
        self.vec.close()
        super().close()


def _scalar(v):
    """numpy scalar -> Python; NaN lap times ("no lap yet") -> None so infos compare equal (gymnasium check_env)."""
    x = v.item()
    return None if isinstance(x, float) and math.isnan(x) else x


def _single(infos: dict) -> dict:
    out = {}
    for k, v in infos.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            out[k] = {kk: _scalar(vv[0]) for kk, vv in v.items()}
        elif k in ("final_obs",):
            out[k] = v[0]
        else:
            out[k] = _scalar(v[0])
    return out
