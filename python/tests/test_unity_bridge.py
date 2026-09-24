"""Tests against the real bridge player (pytest -m unity). The full-size DoD runs live in scripts/m3_validate.py."""

import numpy as np
import pytest

from racing_rl import paths
from racing_rl.bridge import OnnxPolicy, RacingEnv, UnityVecEnv
from racing_rl.bridge.protocol import FROZEN_ENV_CONFIG_HASH, FROZEN_OBS_LAYOUT_HASH

EXE = paths.BRIDGE_EXE
LOGS = paths.RUNS / "pytest_unity_logs"

pytestmark = [pytest.mark.unity, pytest.mark.skipif(not EXE.is_file(), reason=f"bridge player missing: {EXE}")]


def test_handshake_hashes():
    env = UnityVecEnv(EXE, num_agents=2, log_dir=LOGS)
    try:
        assert env.hello["env_config_hash"] == FROZEN_ENV_CONFIG_HASH
        assert env.hello["obs_layout_hash"] == FROZEN_OBS_LAYOUT_HASH
        assert env.num_envs == 2
    finally:
        env.close()


def test_check_env():
    from gymnasium.utils.env_checker import check_env

    env = RacingEnv(EXE, num_agents=1, log_dir=LOGS)
    check_env(env, skip_render_check=True)
    env.close()


def test_reset_is_reproducible_in_process():
    env = UnityVecEnv(EXE, num_agents=4, log_dir=LOGS)
    acts = np.random.default_rng(3).uniform(-1, 1, (50, 4, 2)).astype(np.float32)
    try:
        runs = []
        for _ in range(2):
            obs, _ = env.reset(seed=77)
            seq = [obs]
            for a in acts:
                seq.append(env.step(a)[0])
            runs.append(np.stack(seq))
        assert np.array_equal(runs[0], runs[1])
    finally:
        env.close()


def test_two_processes_bitwise_equal():
    acts = np.random.default_rng(7).uniform(-1, 1, (300, 8, 2)).astype(np.float32)
    out = []
    for _ in range(2):
        env = UnityVecEnv(EXE, num_agents=8, log_dir=LOGS)
        try:
            obs, _ = env.reset(seed=5)
            rec = [obs]
            for a in acts:
                o, r, te, tr, _ = env.step(a)
                rec += [o, r[:, None].repeat(26, 1), te[:, None].repeat(26, 1), tr[:, None].repeat(26, 1)]
            out.append(np.concatenate(rec).astype(np.float32))
        finally:
            env.close()
    assert np.array_equal(out[0], out[1])


def test_onnx_policy_io_names():
    p = OnnxPolicy(paths.MODELS / "mlagents_baseline_s1.onnx")
    assert "obs_0" in p.input_names and "deterministic_continuous_actions" in p.output_names
    a = p(np.zeros((3, 26), np.float32))
    assert a.shape == (3, 2) and np.all(np.abs(a) <= 1.0)
