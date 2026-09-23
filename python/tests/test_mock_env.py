"""UnityVecEnv against mock_unity (no Unity needed): reset/step/auto-reset/final_obs/desync/timeout/crash."""

import json
import socket

import numpy as np
import pytest

from racing_rl.bridge import (DesyncError, ProtocolMismatchError, RemoteError, UnityCrashedError, UnityTimeoutError,
                              UnityVecEnv)
from racing_rl.bridge.mock_unity import MockUnity
from racing_rl.bridge.protocol import (ERR_HASH_MISMATCH, ERR_TRACK_MISMATCH, HELLO_TRACK_FIELDS, INFO_FIELDS, MSG_CONFIG,
                                       MSG_ERROR, MSG_HELLO, MSG_READY, OBS_DIM, TermReason, layout_hash)
from racing_rl.bridge.transport import FramedSocket, bind_server


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


def test_hello_carries_track_fields():
    """M6: HELLO appends the track fields; the M3 fields and PROTOCOL stay as they were."""
    env, _ = make(2, track_id="Track_B", track_index=1)
    try:
        h = env.hello
        assert set(HELLO_TRACK_FIELDS) <= set(h)
        assert h["protocol"] == 1 and h["env_config_hash"] and h["obs_dim"] == OBS_DIM
        assert h["track_id"] == "Track_B" and h["track_index"] == 1
        assert isinstance(h["track_checkpoints"], int) and h["track_checkpoints"] > 0
        assert h["track_length_m"] > 0 and h["track_half_width"] > 0
        assert len(h["track_hash"]) == 16
    finally:
        env.close()


def raw_handshake(cfg: dict, **mock_kw) -> tuple[dict, int, dict]:
    """HELLO/CONFIG by hand (UnityVecEnv sends expected_track_id from M6 step 5 on). Returns (hello, reply type, reply)."""
    srv, port = bind_server("127.0.0.1", free_port())
    try:
        MockUnity(port, 2, **mock_kw).start_thread()
        srv.settimeout(10.0)
        conn, _ = srv.accept()
    finally:
        srv.close()
    sock = FramedSocket(conn, 10.0)
    try:
        t, _, payload = sock.recv()
        assert t == MSG_HELLO
        hello = json.loads(bytes(payload))
        sock.send(MSG_CONFIG, 1, json.dumps(cfg).encode())
        t, seq, payload = sock.recv()
        assert seq == 1
        return hello, t, json.loads(bytes(payload))
    finally:
        sock.close()


@pytest.mark.parametrize("cfg, reply, code", [
    ({"expected_track_id": "Track_D", "strict": True}, MSG_READY, None),
    ({"expected_track_id": "", "strict": True}, MSG_READY, None),
    ({"expected_track_id": "Track_A", "strict": True}, MSG_ERROR, ERR_TRACK_MISMATCH),
    ({"expected_track_id": "Track_A", "strict": False}, MSG_READY, None),
    # the track id is checked before the hashes (BridgeProtocol.CheckConfig)
    ({"expected_track_id": "Track_A", "expected_env_config_hash": "0000000000000000", "strict": True}, MSG_ERROR,
     ERR_TRACK_MISMATCH),
    ({"expected_track_id": "Track_D", "expected_env_config_hash": "0000000000000000", "strict": True}, MSG_ERROR,
     ERR_HASH_MISMATCH),
])
def test_config_expected_track_id(cfg, reply, code):
    cfg = {"expected_obs_layout_hash": layout_hash(), **cfg}
    hello, t, body = raw_handshake(cfg, track_id="Track_D", track_index=3)
    assert hello["track_id"] == "Track_D"
    assert t == reply, body
    if code is not None:
        assert body["code"] == code and body["fatal"] is True


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
