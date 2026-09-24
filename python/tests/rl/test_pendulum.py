"""
DoD 9 (optional, pytest -m slow): Pendulum-v1, 1M steps, mean deterministic return ≥ −250.
Actions are rescaled [−1, 1] → [−2, 2]; rewards are scaled ×0.1 for training only (evaluation uses raw returns).
"""

import time

import gymnasium
import numpy as np
import pytest

from racing_rl.rl import ActorCritic, get_preset, train

pytestmark = pytest.mark.slow

TOTAL = 1_000_000
N_ENVS = 16


def _make(train_env: bool):
    def thunk():
        env = gymnasium.make("Pendulum-v1")
        env = gymnasium.wrappers.RescaleAction(env, -1.0, 1.0)  # a ∈ [−1, 1] → torque 2a
        if train_env:
            env = gymnasium.wrappers.TransformReward(env, lambda r: 0.1 * r)
        return env
    return thunk


def _evaluate(ac: ActorCritic, episodes: int = 20, seed: int = 10_000) -> float:
    returns = []
    for k in range(episodes):
        env = _make(False)()
        obs, _ = env.reset(seed=seed + k)
        total, done = 0.0, False
        while not done:
            a = ac.act_deterministic(np.asarray(obs, np.float32)[None])[0]
            obs, r, term, trunc, _ = env.step(a)
            total += float(r)
            done = term or trunc
        returns.append(total)
        env.close()
    return float(np.mean(returns))


def test_pendulum_1m():
    cfg = get_preset("default", lr=1e-3, lr_end=0.0, gamma=0.95, gae_lambda=0.95, epochs=10, rollout_length=256,
                     minibatch_size=512, ent_coef=0.0, ent_coef_end=0.0, clip_vloss=False, log_std_init=-0.5)
    envs = gymnasium.vector.SyncVectorEnv([_make(True) for _ in range(N_ENVS)],
                                          autoreset_mode=gymnasium.vector.AutoresetMode.SAME_STEP)
    ac = ActorCritic.from_config(cfg, obs_dim=3, act_dim=1)
    t0 = time.perf_counter()
    curve = []

    def log(m):
        if int(m["update"]) % 20 == 0:
            curve.append((int(m["global_step"]), round(10.0 * m["ep_return_mean"], 1)))

    hist = train(envs, cfg, total_steps=TOTAL, ckpt_dir=None, seed=1, ac=ac, log_fn=log)
    envs.close()
    ret = _evaluate(ac)
    print(f"\nDoD-9 Pendulum: {len(hist)} updates, {time.perf_counter() - t0:.0f}s, deterministic mean return "
          f"{ret:.1f}; training curve (step, raw ep return): {curve}")
    assert not any(m["nan_skipped"] for m in hist)
    assert ret >= -250.0
