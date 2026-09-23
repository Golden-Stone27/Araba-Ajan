"""
DoD 7: learning sanity on TargetReachVecEnv (26 dims, target in dims 0–1, r = −‖a − target‖²).
Approved test config: default preset with lr 1e-3, N = 64, T = 64, minibatch 256. Within 30 updates the mean reward
of the DETERMINISTIC policy (μ) on fresh samples must exceed −0.05, in < 60 s on CPU.
"""

import time

import numpy as np

from racing_rl.rl import ActorCritic, PPOTrainer, RolloutBuffer, TargetReachVecEnv, collect_rollout, get_preset

UPDATES = 30
THRESHOLD = -0.05


def _reward(a: np.ndarray, target: np.ndarray) -> float:
    return float(-np.sum((np.clip(a, -1, 1) - target) ** 2, axis=1).mean())


def test_target_reach_learns():
    t0 = time.perf_counter()
    cfg = get_preset("default", lr=1e-3, rollout_length=64, minibatch_size=256)
    env = TargetReachVecEnv(64, episode_len=50, seed=0)
    ac = ActorCritic.from_config(cfg)
    trainer = PPOTrainer(ac, cfg, seed=0)
    buf = RolloutBuffer(64, 64)
    fresh = np.random.default_rng(12345).uniform(-1, 1, (8192, 26)).astype(np.float32)
    obs, _ = env.reset(seed=0)
    det, sto = [], []
    for k in range(UPDATES):
        obs, last_v = collect_rollout(env, ac, buf, obs, cfg)
        buf.compute_gae(last_v, cfg.gamma, cfg.gae_lambda)
        m = trainer.update(buf, k / UPDATES)
        assert m["nan_skipped"] == 0.0
        det.append(_reward(ac.act_deterministic(fresh), fresh[:, :2]))
        sto.append(_reward(ac.act_full(fresh, update_rms=False)[0], fresh[:, :2]))
    wall = time.perf_counter() - t0
    first = next((i + 1 for i, r in enumerate(det) if r > THRESHOLD), None)
    print(f"\nDoD-7 deterministic: {[round(r, 4) for r in det]}\nDoD-7 stochastic: {[round(r, 4) for r in sto]}"
          f"\nDoD-7 first update > {THRESHOLD}: {first}; final det {det[-1]:.4f} sto {sto[-1]:.4f}; wall {wall:.1f}s")
    assert det[-1] > THRESHOLD, det
    assert first is not None and first <= UPDATES
    assert wall < 60.0
