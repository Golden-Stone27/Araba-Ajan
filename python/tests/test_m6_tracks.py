"""
M6 step 5 on mock_unity (no Unity needed): the exported track catalog, track name resolution, UnityVecEnv(track=),
MultiUnityVecEnv(tracks=), the --track flags of train_ppo / evaluate, checkpoint extra.tracks and the catalog-based
checkpoint check (contracts C0.20).
"""

import json
import socket
import sys
from pathlib import Path

import numpy as np
import pytest

from racing_rl.bridge import (MultiUnityVecEnv, ProtocolMismatchError, UnityVecEnv, UnknownTrackError, composite_hash,
                              load_catalog, resolve_track)
from racing_rl.bridge.mock_unity import MockUnity
from racing_rl.bridge.protocol import ACT_DIM, FROZEN_ENV_CONFIG_HASH, FROZEN_OBS_LAYOUT_HASH, OBS_DIM
from racing_rl.bridge.tracks import MULTI_PREFIX, training_tracks
from racing_rl.rl import ActorCritic, load_checkpoint
from racing_rl.rl.config import presets
from racing_rl.rl.policy import CheckpointMismatchError, save_checkpoint
from racing_rl.train import evaluate as ev_cli
from racing_rl.train import train_ppo

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# contracts C0.20: env_config_hash and track_hash (BridgeTrackTests) per catalog track, frozen procedural references
FROZEN_TRACKS = {
    "Track_A": (0, "Benchmark", "90240ee2b1a58b5b", "0f2fbf3481a1a24f", 114),
    "Track_B": (1, "Technical", "038a104393cbfb72", "6ce4f5d0b0d07eda", 111),
    "Track_C": (2, "Speedway", "0e099647315ed638", "ff0ea58517167d6f", 126),
    "Track_D": (3, "Elevation", "dbce4b7f772fc7b4", "04f6036ec025016a", 109),
}
FROZEN_PROC = {"proc:0": "00724614a3c71752", "proc:1": "860ce2361684922d", "proc:7": "59966c9f08fc1027",
               "proc:1000": "7b4353dd8647e3d3"}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def mock_for(port: int, n: int, track: str, **kw) -> MockUnity:
    """A mock that reports `track` with its catalog hashes (the kinematics stay the mock circle)."""
    spec = resolve_track(track)
    e = spec.entry
    args = dict(track_id=spec.id, track_index=spec.index)
    if e is not None:
        args.update(env_config_hash=e.env_config_hash, track_hash=e.track_hash)
    args.update(kw)
    m = MockUnity(port, n, **args)
    m.start_thread()
    return m


def env_on(mock_kw: dict, n: int = 2, **env_kw):
    port = free_port()
    mock = mock_for(port, n, **mock_kw) if "track" in mock_kw else MockUnity(port, n, **mock_kw)
    if "track" not in mock_kw:
        mock.start_thread()
    env = UnityVecEnv(None, num_agents=n, port=port, step_timeout_s=10.0, launch_timeout_s=10.0, **env_kw)
    return env, mock


def make_checkpoint(path: Path, env_hash: str, extra: dict | None = None, obs_hash: str = FROZEN_OBS_LAYOUT_HASH) -> Path:
    cfg = presets["parity"]
    return save_checkpoint(path, ActorCritic.from_config(cfg, OBS_DIM, ACT_DIM), None, cfg, 0, env_hash, obs_hash, extra)


# ------------------------------------------------------------------------------------------------ catalog
def test_catalog_matches_contracts():
    cat = load_catalog()
    assert cat.obs_layout_hash == FROZEN_OBS_LAYOUT_HASH and cat.generator_version == 1
    assert cat.ids[:4] == list(FROZEN_TRACKS)
    for t in cat.tracks[:4]:
        index, profile, env_hash, track_hash, gates = FROZEN_TRACKS[t.id]
        assert (t.index, t.profile, t.env_config_hash, t.track_hash, t.checkpoints) == (index, profile, env_hash,
                                                                                         track_hash, gates)
        assert t.half_width == t.width / 2 and t.elevation == (t.id == "Track_D")
    assert {p.id: p.env_config_hash for p in cat.procedural_refs} == FROZEN_PROC
    assert all(p.index == -1 and p.profile == "Procedural" for p in cat.procedural_refs)


@pytest.mark.parametrize("name, tid, index, env_hash", [
    ("Track_A", "Track_A", 0, FROZEN_ENV_CONFIG_HASH),
    ("Track_B", "Track_B", 1, "038a104393cbfb72"),
    ("TrackDefinition_C", "Track_C", 2, "0e099647315ed638"),  # asset names resolve like TrackCatalog.TryResolve
    ("proc:7", "proc:7", -1, "59966c9f08fc1027"),
    ("proc:007", "proc:7", -1, "59966c9f08fc1027"),  # canonical id = ProceduralTrackGenerator.Name(seed)
    ("proc:42", "proc:42", -1, None),  # not a frozen reference: the hash is recorded from HELLO
])
def test_resolve_track(name, tid, index, env_hash):
    spec = resolve_track(name)
    assert (spec.id, spec.index, spec.expected_env_hash) == (tid, index, env_hash)
    assert spec.procedural == (index < 0)


@pytest.mark.parametrize("name", ["Track_Z", "track_a", "", "proc:", "proc:-1", "proc:+7", "proc: 7", "proc:abc",
                                  "proc:12345678901234567890", "proc:9223372036854775808"])
def test_resolve_track_rejects(name):
    with pytest.raises(UnknownTrackError):
        resolve_track(name)


def test_composite_hash():
    a, b = FROZEN_TRACKS["Track_A"][2], FROZEN_TRACKS["Track_B"][2]
    assert composite_hash({"Track_A": a}) == a  # one track keeps its plain hash
    mixed = composite_hash({"Track_A": a, "Track_B": b})
    assert mixed.startswith(MULTI_PREFIX) and len(mixed) == len(MULTI_PREFIX) + 16
    assert composite_hash({"Track_B": b, "Track_A": a}) == mixed  # order-free
    assert composite_hash({"Track_A": a, "Track_C": FROZEN_TRACKS["Track_C"][2]}) != mixed
    with pytest.raises(ValueError):
        composite_hash({})


def test_training_tracks():
    a, b, d = (FROZEN_TRACKS[t][2] for t in ("Track_A", "Track_B", "Track_D"))
    assert training_tracks(a, None) == {"Track_A": a}  # M5 checkpoints (no extra.tracks)
    assert training_tracks(FROZEN_PROC["proc:7"], {}) == {"proc:7": FROZEN_PROC["proc:7"]}
    assert training_tracks(b, {"Track_B": b}) == {"Track_B": b}
    mixed = {"Track_A": a, "Track_D": d}
    assert training_tracks(composite_hash(mixed), mixed) == mixed
    assert training_tracks("0123456789abcdef", {"proc:42": "0123456789abcdef"}) == {"proc:42": "0123456789abcdef"}
    for env_hash, recorded in [("deadbeefdeadbeef", None),  # unknown environment (e.g. changed reward)
                               (b, {"Track_B": a}),  # recorded hash disagrees with the catalog
                               (composite_hash(mixed), {"Track_A": a}),  # composite does not match
                               (a, {"Track_Z": a})]:  # unknown track
        with pytest.raises(ValueError):
            training_tracks(env_hash, recorded)


# ------------------------------------------------------------------------------------------------ UnityVecEnv
def test_track_none_is_the_m5_handshake():
    env, mock = env_on({}, n=3)  # default mock: track fields of a pre-catalog "mock_circle"
    try:
        assert "expected_track_id" not in mock.config
        assert mock.config["expected_env_config_hash"] == FROZEN_ENV_CONFIG_HASH
        assert env.track_info["track_id"] == "mock_circle" and env.track_info["track_index"] == -1
        _, info = env.reset(seed=1)
        assert info["track_index"].tolist() == [-1, -1, -1] and not info["track_index"].flags.writeable
    finally:
        env.close()


def test_track_catalog_track():
    env, mock = env_on({"track": "Track_B"}, n=2, track="Track_B")
    try:
        assert mock.config["expected_track_id"] == "Track_B"
        assert mock.config["expected_env_config_hash"] == "038a104393cbfb72"
        ti = env.track_info
        assert (ti["track_id"], ti["track_index"], ti["profile"], ti["env_config_hash"]) == (
            "Track_B", 1, "Technical", "038a104393cbfb72")
        assert env.extra_args == []  # Editor / mock mode: no command line; CONFIG guards the track
        _, info = env.reset(seed=1)
        assert info["track_index"].tolist() == [1, 1]
        _, _, _, _, info = env.step(np.zeros((2, ACT_DIM), np.float32))
        assert info["track_index"].tolist() == [1, 1] and info["_track_index"].all()
    finally:
        env.close()


@pytest.mark.parametrize("mock_kw, env_track, needle", [
    ({"track": "Track_C"}, "Track_B", "track_id Track_C != Track_B"),
    ({"track": "Track_B", "env_config_hash": FROZEN_ENV_CONFIG_HASH}, "Track_B", "env_config_hash"),
    ({"track": "Track_B", "track_index": 2}, "Track_B", "track_index 2 != 1"),
    ({"track": "Track_B", "track_hash": "0000000000000000"}, "Track_B", "track_hash"),
    ({"track": "proc:7", "env_config_hash": "0000000000000000"}, "proc:7", "env_config_hash"),  # frozen reference
    ({"track": "Track_A", "track_fields": False}, "Track_A", "predates M6"),
])
def test_track_mismatch_is_refused(mock_kw, env_track, needle):
    with pytest.raises(ProtocolMismatchError, match=needle):
        env_on(mock_kw, track=env_track)


def test_track_mismatch_warns_when_not_strict():
    with pytest.warns(UserWarning, match="track_id Track_C != Track_B"):
        env, mock = env_on({"track": "Track_C"}, track="Track_B", strict=False, expected_env_hash=None)
    try:
        assert mock.config["expected_track_id"] == "Track_B" and env.track_info["track_id"] == "Track_C"
    finally:
        env.close()


def test_unfrozen_proc_seed_records_hello_hash():
    env, mock = env_on({"track": "proc:42", "env_config_hash": "0123456789abcdef"}, track="proc:42")
    try:
        assert mock.config["expected_env_config_hash"] == "" and mock.config["expected_track_id"] == "proc:42"
        ti = env.track_info
        assert (ti["track_id"], ti["track_index"], ti["profile"], ti["env_config_hash"]) == (
            "proc:42", -1, "Procedural", "0123456789abcdef")
    finally:
        env.close()


def test_pre_m6_build_without_track():
    env, _ = env_on({"track_fields": False})
    try:
        assert env.track_info is None
        _, info = env.reset(seed=1)
        assert "track_index" not in info
    finally:
        env.close()


def test_unknown_track_fails_before_launch():
    with pytest.raises(UnknownTrackError):
        UnityVecEnv(None, num_agents=2, port=free_port(), launch_timeout_s=0.5, track="Track_Z")


def test_player_gets_track_name(monkeypatch):
    def fake_launch(self):
        self.hello = {"num_agents": 2}

    monkeypatch.setattr(UnityVecEnv, "_launch", fake_launch)
    exe = sys.executable  # any existing file; _launch is stubbed
    assert UnityVecEnv(exe, num_agents=2, track="proc:007").extra_args == ["-trackName", "proc:7"]
    assert UnityVecEnv(exe, num_agents=2).extra_args == []
    with pytest.raises(ValueError, match="not both"):
        UnityVecEnv(exe, num_agents=2, track="Track_B", extra_args=["-trackName", "Track_B"])


# ------------------------------------------------------------------------------------------------ MultiUnityVecEnv
def test_multi_env_runs_one_track_per_process():
    base, n = free_port(), 2
    tracks = ["Track_A", "Track_B"]
    for k in range(3):
        mock_for(base + 25 * k, n, tracks[k % 2])
    with pytest.warns(UserWarning, match="unequal"):
        envs = MultiUnityVecEnv(None, 3, n, base, step_timeout_s=10.0, launch_timeout_s=10.0, tracks=tracks)
    try:
        assert envs.track_ids == ["Track_A", "Track_B", "Track_A"]
        assert envs.env_hashes == {"Track_A": FROZEN_ENV_CONFIG_HASH, "Track_B": "038a104393cbfb72"}
        assert envs.env_config_hash == composite_hash(envs.env_hashes)
        assert envs.env_config_hash.startswith(MULTI_PREFIX)
        _, info = envs.reset(seed=3)
        assert info["track_index"].tolist() == [0, 0, 1, 1, 0, 0]
        _, _, _, _, info = envs.step(np.zeros((6, ACT_DIM), np.float32))
        assert info["track_index"].tolist() == [0, 0, 1, 1, 0, 0]
    finally:
        envs.close()


def test_multi_env_single_track_hash():
    base, n = free_port(), 2
    for k in range(2):
        mock_for(base + 25 * k, n, "Track_D")
    envs = MultiUnityVecEnv(None, 2, n, base, step_timeout_s=10.0, launch_timeout_s=10.0, tracks=["Track_D"])
    try:
        assert envs.track_ids == ["Track_D", "Track_D"] and envs.env_config_hash == "dbce4b7f772fc7b4"
    finally:
        envs.close()


def test_multi_env_argument_checks():
    with pytest.raises(ValueError, match="at least as many processes"):
        MultiUnityVecEnv(None, 1, 2, free_port(), tracks=["Track_A", "Track_B"])
    with pytest.raises(ValueError, match="own catalog hash"):
        MultiUnityVecEnv(None, 2, 2, free_port(), FROZEN_ENV_CONFIG_HASH, tracks=["Track_A", "Track_B"])
    with pytest.raises(UnknownTrackError):
        MultiUnityVecEnv(None, 2, 2, free_port(), tracks=["Track_A", "Track_Z"])


# ------------------------------------------------------------------------------------------------ train_ppo / evaluate
def run_train(run_dir, ports, eval_port, *extra, seed=1):
    args = ["--run-dir", str(run_dir), "--exe", "", "--seed", str(seed), "--procs", str(len(ports)), "--agents", "4",
            "--base-port", str(ports[0]), "--eval-port", str(eval_port), "--total-steps", "256",
            "--eval-every", "2", "--ckpt-every", "1", "--milestones", "64", "--no-tensorboard",
            "--config", str(CONFIGS / "ppo_parity.yaml"),
            "--set", "rollout_total=null", "rollout_length=8", "minibatch_size=32", *extra]
    return train_ppo.main(args)


def start_mocks(ports, tracks, eval_port, eval_track):
    for k, p in enumerate(ports):
        mock_for(p, 4, tracks[k % len(tracks)], max_episode_decisions=12)
    mock_for(eval_port, 20, eval_track)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_train_single_track(tmp_path):
    base, ev = free_port(), free_port()
    ports = [base, base + 25]
    start_mocks(ports, ["Track_B"], ev, "Track_B")
    run_dir = tmp_path / "b"
    assert run_train(run_dir, ports, ev, "--track", "TrackDefinition_B") == 0
    b = "038a104393cbfb72"
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["env_config_hash"] == b and meta["tracks"] == {"Track_B": b}
    assert meta["run"]["tracks"] == ["Track_B"] and meta["run"]["eval_track"] == "Track_B"
    ck = load_checkpoint(run_dir / "checkpoints" / "ckpt_256.pt", b)
    assert ck["extra"]["tracks"] == {"Track_B": b} and ck["format"] == "racing_rl.ppo/v1"
    assert all(r["track"] == "Track_B" for r in jsonl(run_dir / "eval.jsonl"))
    _, name, _, tracks = ev_cli.load_policy_with_tracks(f"torch:{run_dir / 'best.pt'}")
    assert name == "custom-ppo" and tracks == {"Track_B": b}


def test_train_mixed_tracks_and_resume(tmp_path):
    base, ev = free_port(), free_port()
    ports = [base, base + 25]
    tracks = ["Track_A", "Track_D"]
    start_mocks(ports, tracks, ev, "Track_A")
    run_dir = tmp_path / "mix"
    assert run_train(run_dir, ports, ev, "--track", "Track_A", "--track", "Track_D", "--max-updates", "2") == 0
    expected = {"Track_A": FROZEN_ENV_CONFIG_HASH, "Track_D": "dbce4b7f772fc7b4"}
    multi = composite_hash(expected)
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["env_config_hash"] == multi and meta["tracks"] == expected

    start_mocks(ports, tracks, ev, "Track_A")
    assert train_ppo.main(["--run-dir", str(run_dir), "--exe", "", "--resume", "--no-tensorboard"]) == 0
    s = json.loads((run_dir / "summary.json").read_text())
    assert s["status"] == "finished" and s["global_step"] == 256
    last = load_checkpoint(run_dir / "checkpoints" / "ckpt_256.pt", multi)
    assert last["extra"]["tracks"] == expected
    assert training_tracks(last["env_config_hash"], last["extra"]["tracks"]) == expected
    assert ev_cli.load_policy_with_tracks(f"torch:{run_dir / 'best.pt'}")[3] == expected


def test_run_config_tracks(tmp_path):
    y = tmp_path / "mix.yaml"
    y.write_text("ppo:\n  preset: parity\nrun:\n  tracks: [Track_C, proc:007]\n", encoding="utf-8")
    _, run = train_ppo.load_run_config(str(y))
    run = {**train_ppo.RUN_DEFAULTS, **run}
    train_ppo.normalize_tracks(run)
    assert run["tracks"] == ["Track_C", "proc:7"] and run["eval_track"] == "Track_C"
    run = {"tracks": "Track_B", "eval_track": "Track_A"}
    train_ppo.normalize_tracks(run)
    assert run == {"tracks": ["Track_B"], "eval_track": "Track_A"}
    run = dict(train_ppo.RUN_DEFAULTS)
    train_ppo.normalize_tracks(run)
    assert run["tracks"] is None and run["eval_track"] is None  # M5 run


def test_train_unknown_track_fails_before_unity(tmp_path):
    with pytest.raises(UnknownTrackError):
        run_train(tmp_path / "z", [free_port()], free_port(), "--track", "Track_Z")
    assert not (tmp_path / "z").exists()


def test_evaluate_cli_zero_shot_report(tmp_path):
    ck = make_checkpoint(tmp_path / "a.pt", FROZEN_ENV_CONFIG_HASH)  # an M5-style checkpoint (Track_A, no extra)
    port, out = free_port(), tmp_path / "eval_b.json"
    mock_for(port, 3, "Track_B", max_episode_decisions=30)
    ev_cli.main(["--policy", f"torch:{ck}", "--train-seed", "1", "--out", str(out), "--exe", "", "--track", "Track_B",
                 "--port", str(port), "--episodes", "3"])
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["env_config_hash"] == "038a104393cbfb72"  # the evaluated track's environment
    assert rep["track"]["track_id"] == "Track_B" and rep["track"]["track_index"] == 1
    assert rep["train_tracks"] == {"Track_A": FROZEN_ENV_CONFIG_HASH}
    assert rep["per_seed"][0]["seed"] == 1 and "track" not in rep["per_seed"][0]  # per_seed keeps the M5 keys


def test_load_policy_checks(tmp_path):
    ok = make_checkpoint(tmp_path / "d.pt", "dbce4b7f772fc7b4")
    assert ev_cli.load_policy_with_tracks(f"torch:{ok}")[3] == {"Track_D": "dbce4b7f772fc7b4"}
    assert len(ev_cli.load_policy(f"torch:{ok}")) == 3  # M5 signature unchanged
    for name, env_hash, obs_hash in [("x.pt", "deadbeefdeadbeef", FROZEN_OBS_LAYOUT_HASH),
                                     ("y.pt", FROZEN_ENV_CONFIG_HASH, "0000000000000000")]:
        bad = make_checkpoint(tmp_path / name, env_hash, obs_hash=obs_hash)
        with pytest.raises(CheckpointMismatchError):
            ev_cli.load_policy(f"torch:{bad}")
