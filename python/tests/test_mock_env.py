"""UnityVecEnv against mock_unity (no Unity needed): reset/step/auto-reset/final_obs/desync/timeout/crash."""

import socket

import numpy as np
import pytest

from racing_rl.bridge import (DesyncError, ProtocolMismatchError, RemoteError, UnityCrashedError, UnityTimeoutError,
                              UnityVecEnv)
from racing_rl.bridge.mock_unity import MockUnity
from racing_rl.bridge.protocol import INFO_FIELDS, OBS_DIM, TermReason


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make(n=4, step_timeout_s=10.0, **mock_kw):
    port = free_port()
    mock = MockUnity(port, n, **mock_kw)
    mock.start_thread()
    env = UnityVecEnv(None, num_agents=n, port=port, step_timeout_s=step_timeout_s, launch_timeout_s=10.0)
    return env, mock


def test_reset_and_spaces():
    env, _ = make(4)
    try:
        obs, info = env.reset(seed=3)
        assert obs.shape == (4, OBS_DIM) and obs.dtype == np.float32
        assert env.single_observation_space.contains(obs[0])
        assert set(INFO_FIELDS) <= set(info)
        obs2, _ = env.reset(seed=3)
        assert np.array_equal(obs, obs2)
        obs3, _ = env.reset(seed=4)
        assert not np.array_equal(obs, obs3)
    finally:
        env.close()


def test_step_autoreset_final_obs_and_reward_consistency():
    env, _ = make(4, max_episode_decisions=60)
    try:
        env.reset(seed=1)
        rng = np.random.default_rng(7)
        ep_sum = np.zeros(4)
        checked = 0
        for _ in range(400):
            a = rng.uniform(-1, 1, (4, 2)).astype(np.float32)
            a[:, 1] = np.abs(a[:, 1])
            obs, rew, term, trunc, info = env.step(a)
            ep_sum += rew
            done = term | trunc
            assert np.array_equal(info["_final_obs"], done)
            for i in np.flatnonzero(done):
                assert np.any(info["final_obs"][i] != 0)
                assert info["final_info"]["term_reason"][i] != TermReason.NONE
                assert abs(ep_sum[i] - info["ep_return"][i]) <= 1e-4
                if trunc[i]:
                    assert info["term_reason"][i] == TermReason.TIME_LIMIT
                    assert info["ep_decisions"][i] == 60
                ep_sum[i] = 0.0
                checked += 1
            assert not np.any(info["final_obs"][~done])
        assert checked >= 4
        assert env.requests_sent == env.replies_received
    finally:
        env.close()


def test_desync_is_fatal():
    env, _ = make(2, desync_at_step=3)
    try:
        env.reset(seed=0)
        env.step(np.zeros((2, 2)))
        env.step(np.zeros((2, 2)))
        with pytest.raises(DesyncError):
            env.step(np.zeros((2, 2)))
        assert not env.connected
    finally:
        env.close()


def test_timeout():
    env, _ = make(2, step_timeout_s=0.5, hang_at_step=2, hang_s=3.0)
    try:
        env.reset(seed=0)
        env.step(np.zeros((2, 2)))
        with pytest.raises(UnityTimeoutError):
            env.step(np.zeros((2, 2)))
        assert not env.connected
    finally:
        env.close()


def test_crash():
    env, _ = make(2, crash_at_step=2)
    try:
        env.reset(seed=0)
        env.step(np.zeros((2, 2)))
        with pytest.raises(UnityCrashedError):
            env.step(np.zeros((2, 2)))
    finally:
        env.close()


def test_nan_action_rejected_in_python():
    env, _ = make(2)
    try:
        env.reset(seed=0)
        with pytest.raises(ValueError):
            env.step(np.array([[np.nan, 0], [0, 0]], np.float32))
        env.step(np.zeros((2, 2)))  # still usable: nothing was sent
    finally:
        env.close()


def test_hash_mismatch():
    port = free_port()
    MockUnity(port, 2, env_config_hash="deadbeefdeadbeef").start_thread()
    with pytest.raises(ProtocolMismatchError):
        UnityVecEnv(None, num_agents=2, port=port, launch_timeout_s=10.0)


def test_hash_mismatch_detected_by_unity_side():
    """strict CONFIG with a hash Unity does not have → ERROR(HASH_MISMATCH) → ProtocolMismatchError."""
    port = free_port()
    MockUnity(port, 2).start_thread()
    with pytest.raises(ProtocolMismatchError):
        UnityVecEnv(None, num_agents=2, port=port, launch_timeout_s=10.0, expected_env_hash="0000000000000000", strict=True)


def test_remote_error_type():
    assert issubclass(RemoteError, Exception)


def test_eval_grid_finishes():
    env, _ = make(3)
    try:
        obs, _ = env.reset(seed=1000, options={"start_mode": 1, "max_laps": 1})
        for _ in range(2000):
            # P-controller on the mock's kinematics: feed-forward curvature + heading + lateral terms.
            steer = obs[:, 15] * 50.0 / (0.8 * 60.0) + 2.0 * obs[:, 18] - 1.0 * obs[:, 22]
            a = np.stack([np.clip(steer, -1, 1), np.full(3, 0.2)], axis=1).astype(np.float32)
            obs, rew, term, trunc, info = env.step(a)
            if trunc.any():
                i = int(np.flatnonzero(trunc)[0])
                assert info["term_reason"][i] == TermReason.FINISHED
                assert info["laps"][i] == 1 and info["lap_completed"][i] == 1
                assert np.isfinite(info["last_lap_s"][i])
                return
            assert not term.any(), info["term_reason"]
        pytest.fail("no episode finished")
    finally:
        env.close()
