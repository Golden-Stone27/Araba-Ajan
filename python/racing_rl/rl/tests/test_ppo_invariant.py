"""
DoD 5: after a real collect_rollout, the first minibatch of the first epoch sees ρ ≡ 1 (±1e-6), clip_frac = 0,
approx_kl ≈ 0. A negative control re-normalises the stored obs with the end-of-rollout statistics (the classic
collection/evaluation normalisation mismatch) and must break the invariant.
"""

import copy

import numpy as np
import torch

from racing_rl.rl import ActorCritic, PPOConfig, PPOTrainer, RolloutBuffer, TargetReachVecEnv, collect_rollout


class _Recorder:
    """Wraps a VectorEnv and keeps every observation handed to the policy."""

    def __init__(self, env):
        self.env, self.num_envs, self.obs = env, env.num_envs, []

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.obs = [obs]
        return obs, info

    def step_async(self, a):
        self.env.step_async(a)

    def step_wait(self):
        out = self.env.step_wait()
        self.obs.append(out[0])
        return out


def _setup(parity: bool = False):
    cfg = PPOConfig(rollout_length=32, minibatch_size=128)
    if parity:
        cfg = cfg.replace(activation="silu", action_transform="mlagents_scale", log_std_init=0.0, epochs=3)
    env = _Recorder(TargetReachVecEnv(16, episode_len=10, seed=3))
    ac = ActorCritic.from_config(cfg)
    buf = RolloutBuffer(32, 16)
    obs, _ = env.reset(seed=3)
    obs, last_v = collect_rollout(env, ac, buf, obs, cfg)
    buf.compute_gae(last_v, cfg.gamma, cfg.gae_lambda)
    return cfg, env, ac, buf


def test_first_minibatch_ratio_is_one():
    for parity in (False, True):
        cfg, _, ac, buf = _setup(parity)
        m = PPOTrainer(ac, cfg, seed=0).update(buf, 0.0)
        assert m["nan_skipped"] == 0.0
        assert m["first_mb_ratio_maxdev"] <= 1e-6, m["first_mb_ratio_maxdev"]
        assert m["first_mb_clip_frac"] == 0.0
        assert abs(m["first_mb_approx_kl"]) <= 1e-10
        assert m["approx_kl"] > 0.0  # later minibatches do move the policy


def test_normalisation_mismatch_is_detected():
    cfg, env, ac, buf = _setup()
    raw = np.stack(env.obs[:32])  # [T, N, 26] raw observations the policy saw
    buf_bad = copy.deepcopy(buf)
    buf_bad.obs.copy_(torch.from_numpy(ac.obs_rms.normalize(raw, cfg.obs_clip).astype(np.float32)))
    m = PPOTrainer(copy.deepcopy(ac), cfg, seed=0).update(buf_bad, 0.0)
    assert m["first_mb_ratio_maxdev"] > 1e-3
