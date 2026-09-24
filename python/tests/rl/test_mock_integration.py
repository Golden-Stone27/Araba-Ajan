"""
Extra-b: train() over UnityVecEnv backed by mock_unity (real TCP bridge, no Unity), then the C0.10 bridge evaluator
(run_eval + per_seed_report, 20 agents) drives TorchPolicy unchanged; eval_fn → best.pt ranking path.
"""

import json
import socket

import numpy as np

from racing_rl.bridge import UnityVecEnv
from racing_rl.bridge.evaluate import per_seed_report, report, run_eval
from racing_rl.bridge.mock_unity import MockUnity
from racing_rl.bridge.protocol import FROZEN_ENV_CONFIG_HASH, layout_hash
from racing_rl.rl import ActorCritic, PPOConfig, TorchPolicy, load_checkpoint, train

PER_SEED_KEYS = {"seed", "completion_rate", "flying_lap_median_s", "flying_lap_best_s", "lap1_median_s",
                 "mean_speed_mps", "term_reasons", "sector_times_s"}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def mock_env(n: int, **kw) -> UnityVecEnv:
    port = free_port()
    MockUnity(port, n, **kw).start_thread()
    return UnityVecEnv(None, num_agents=n, port=port, step_timeout_s=30.0, launch_timeout_s=10.0)


def test_train_and_bridge_eval_on_mock(tmp_path):
    cfg = PPOConfig(rollout_length=16, minibatch_size=32)
    env = mock_env(4, max_episode_decisions=20)
    eval_env = mock_env(20)
    evals = []

    def eval_fn(policy):
        rep = per_seed_report(run_eval(eval_env, policy), train_seed=1)
        evals.append(rep)
        return rep

    try:
        ac = ActorCritic.from_config(cfg)
        hist = train(env, cfg, total_steps=2 * 16 * 4, ckpt_dir=tmp_path, eval_fn=eval_fn, eval_every=1,
                     ckpt_every=1, seed=1, ac=ac)
        assert len(hist) == 2 and hist[-1]["global_step"] == 128.0
        assert all(m["nan_skipped"] == 0.0 and np.isfinite(m["policy_loss"]) for m in hist)
        assert sum(m["episodes"] for m in hist) >= 4  # max_episode_decisions 20 < 2·16: every agent finished one
        assert sorted(p.name for p in tmp_path.glob("*.pt")) == ["best.pt", "ckpt_128.pt", "ckpt_64.pt"]
        assert len(evals) == 2 and hist[0]["eval_best"] == 1.0

        # hashes come from the bridge HELLO; best.pt carries the eval dict that ranked it
        best = load_checkpoint(tmp_path / "best.pt", FROZEN_ENV_CONFIG_HASH, layout_hash())
        assert best["extra"]["eval"]["episodes"] == 20

        # the bridge evaluator drives TorchPolicy as-is (Policy protocol = __call__)
        policy = TorchPolicy(ac)
        rms_count = ac.obs_rms.count
        eps = run_eval(eval_env, policy)
        rep = per_seed_report(eps, train_seed=1)
        assert PER_SEED_KEYS <= set(rep) and rep["episodes"] == 20
        assert 0.0 <= rep["completion_rate"] <= 1.0 and sum(rep["term_reasons"].values()) == 20
        assert ac.obs_rms.count == rms_count  # eval never updates the statistics
        full = report(eval_env, [rep], "custom-ppo")
        assert full["schema"] == "race-benchmark/v1" and full["env_config_hash"] == FROZEN_ENV_CONFIG_HASH
        json.dumps(full)  # C0.10 JSON-serialisable
    finally:
        env.close()
        eval_env.close()
