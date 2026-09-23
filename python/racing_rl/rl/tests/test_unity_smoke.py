"""
Extra-c (pytest -m unity): the real bridge player, 16 agents, 3 PPO updates (default preset) → no NaN, SPS,
checkpoint written; then the C0.10 bridge eval (20 agents) of the trained TorchPolicy → race-benchmark/v1 JSON.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from racing_rl.bridge import UnityVecEnv
from racing_rl.bridge.evaluate import per_seed_report, report, run_eval
from racing_rl.bridge.protocol import FROZEN_ENV_CONFIG_HASH, layout_hash
from racing_rl.rl import ActorCritic, TorchPolicy, get_preset, load_checkpoint, train

REPO = Path(__file__).resolve().parents[4]
EXE = REPO / "Builds" / "RaceEnv" / "RaceEnv.exe"
LOGS = REPO / "runs" / "pytest_unity_logs"
PORT = 6405

pytestmark = [pytest.mark.unity, pytest.mark.skipif(not EXE.is_file(), reason=f"bridge player missing: {EXE}")]


def test_unity_three_updates_and_eval(tmp_path):
    cfg = get_preset("default")
    env = UnityVecEnv(EXE, num_agents=16, port=PORT, log_dir=LOGS)
    try:
        ac = ActorCritic.from_config(cfg)
        t0 = time.perf_counter()
        hist = train(env, cfg, total_steps=3 * 256 * 16, ckpt_dir=tmp_path / "ckpt", seed=1, ac=ac)
        wall = time.perf_counter() - t0
    finally:
        env.close()
    assert len(hist) == 3
    for m in hist:
        assert m["nan_skipped"] == 0.0
        assert all(np.isfinite(m[k]) for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "grad_norm"))
    ckpts = list((tmp_path / "ckpt").glob("ckpt_*.pt"))
    assert [p.name for p in ckpts] == ["ckpt_12288.pt"]
    load_checkpoint(ckpts[0], FROZEN_ENV_CONFIG_HASH, layout_hash())

    eval_env = UnityVecEnv(EXE, num_agents=20, port=PORT + 30, log_dir=LOGS)
    try:
        t1 = time.perf_counter()
        eps = run_eval(eval_env, TorchPolicy(ac))
        rep = report(eval_env, [per_seed_report(eps, train_seed=1)], "custom-ppo",
                     {"wallclock_s": round(time.perf_counter() - t1, 2)})
    finally:
        eval_env.close()
    out = tmp_path / "eval_smoke.json"
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    assert rep["env_config_hash"] == FROZEN_ENV_CONFIG_HASH and rep["per_seed"][0]["episodes"] == 20

    summary = {"train_wall_s": round(wall, 2), "decisions": int(hist[-1]["global_step"]),
               "sps_total": round(hist[-1]["global_step"] / wall, 1),
               "sps_per_update": [round(m["sps"], 1) for m in hist],
               "collect_s": [round(m["time_collect_s"], 2) for m in hist],
               "update_s": [round(m["time_update_s"], 2) for m in hist],
               "ep_return_mean": [m["ep_return_mean"] for m in hist],
               "approx_kl": [m["approx_kl"] for m in hist], "explained_var": [m["explained_var"] for m in hist],
               "eval": {k: rep["per_seed"][0][k] for k in ("completion_rate", "term_reasons", "mean_speed_mps")},
               "eval_wall_s": rep["wallclock_s"], "eval_json": str(out)}
    print("\nM4 unity smoke:", json.dumps(summary, indent=1))
