"""M6 step 5 against the real bridge player (pytest -m unity): per-track smoke runs, mixed-track MultiUnityVecEnv."""

from pathlib import Path

import numpy as np
import pytest

from racing_rl.bridge import MultiUnityVecEnv, RemoteError, StartMode, UnityVecEnv, composite_hash, resolve_track
from racing_rl.bridge.protocol import ERR_UNKNOWN_TRACK, FROZEN_OBS_LAYOUT_HASH

REPO = Path(__file__).resolve().parents[2]
EXE = REPO / "Builds" / "RaceEnv" / "RaceEnv.exe"
LOGS = REPO / "runs" / "pytest_unity_logs"

pytestmark = [pytest.mark.unity, pytest.mark.skipif(not EXE.is_file(), reason=f"bridge player missing: {EXE}")]

DRIVE = np.array([0.0, 0.6], np.float32)  # straight, part throttle


@pytest.mark.parametrize("track", ["Track_A", "Track_B", "Track_C", "Track_D", "proc:7", "proc:42"])
def test_track_smoke(track):
    spec = resolve_track(track)
    env = UnityVecEnv(EXE, num_agents=4, port=6605, log_dir=LOGS, track=track)
    try:
        ti = env.track_info
        assert (ti["track_id"], ti["track_index"]) == (spec.id, spec.index)
        assert env.hello["obs_layout_hash"] == FROZEN_OBS_LAYOUT_HASH
        if spec.entry is not None:  # catalog track or frozen procedural reference: every HELLO field as exported
            e = spec.entry
            assert (ti["env_config_hash"], ti["track_hash"], ti["track_checkpoints"]) == (
                e.env_config_hash, e.track_hash, e.checkpoints)
            assert ti["track_length_m"] == e.length_m and ti["track_half_width"] == e.half_width
            assert ti["profile"] == e.profile
        else:
            assert ti["profile"] == "Procedural" and len(ti["env_config_hash"]) == 16
        obs, info = env.reset(seed=1000, options={"start_mode": int(StartMode.EVAL_GRID), "max_laps": 3})
        assert info["track_index"].tolist() == [spec.index] * 4
        progress = info["progress"].copy()
        for _ in range(50):
            obs, _, term, trunc, info = env.step(np.tile(DRIVE, (4, 1)))
            assert np.isfinite(obs).all()
        assert info["track_index"].tolist() == [spec.index] * 4
        assert (info["progress"] > progress).all()  # the cars moved forward along the track
    finally:
        env.close()


def test_unknown_track_from_the_player():
    """Bypassing the Python-side check (raw flag): the player itself answers UNKNOWN_TRACK instead of HELLO."""
    with pytest.raises(RemoteError) as exc:
        UnityVecEnv(EXE, num_agents=2, port=6605, log_dir=LOGS, extra_args=["-trackName", "Track_Z"])
    assert exc.value.code == ERR_UNKNOWN_TRACK and exc.value.fatal


def test_multi_env_two_tracks():
    envs = MultiUnityVecEnv(EXE, num_processes=2, num_agents=4, base_port=6705, log_dir=LOGS,
                            tracks=["Track_B", "Track_D"])
    try:
        assert envs.track_ids == ["Track_B", "Track_D"]
        assert envs.env_hashes == {"Track_B": "038a104393cbfb72", "Track_D": "dbce4b7f772fc7b4"}
        assert envs.env_config_hash == composite_hash(envs.env_hashes)
        _, info = envs.reset(seed=5)
        assert info["track_index"].tolist() == [1] * 4 + [3] * 4
        for _ in range(20):
            obs, *_ = envs.step(np.tile(DRIVE, (8, 1)))
            assert np.isfinite(obs).all()
    finally:
        envs.close()
