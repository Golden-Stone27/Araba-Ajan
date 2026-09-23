"""
M5 train package on mock_unity (real TCP bridge, no Unity): telemetry aggregation, run config, the train_ppo CLI
end to end (checkpoints, milestones, validation eval → best.pt, TensorBoard/CSV), --resume, crash recovery and the
reproducibility DoD (same config + seed → identical first-update metrics).
"""

import json
import math
import socket
import threading
from pathlib import Path

import numpy as np
import pytest

from racing_rl.bridge.mock_unity import MockUnity
from racing_rl.bridge.protocol import TermReason
from racing_rl.rl import load_checkpoint
from racing_rl.train import train_ppo
from racing_rl.train.telemetry import RaceStats

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
TIMING_KEYS = {"time_collect_s", "time_update_s", "sps", "collect_sps", "wallclock_s", "time_total_s",
               "step_latency_p50_ms", "step_latency_p99_ms", "time_eval_s"}


def free_ports(n: int) -> list[int]:
    socks, ports = [], []
    for _ in range(n):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        socks.append(s)
        ports.append(s.getsockname()[1])
    for s in socks:
        s.close()
    return ports


def infos(n, **over):
    d = {"speed_mps": np.full(n, 10.0, np.float32), "lap_completed": np.zeros(n, np.uint8),
         "last_lap_s": np.full(n, np.nan, np.float32), "term_reason": np.zeros(n, np.uint8),
         "laps": np.zeros(n, np.uint16)}
    d.update(over)
    return d


def test_race_stats_mirrors_mlagents_episode_stats():
    st = RaceStats(4)
    no = np.zeros(4, bool)
    st.on_step(no, no, infos(4), 0.002)
    st.on_step(no, no, infos(4, lap_completed=np.array([1, 0, 0, 1], np.uint8),
                              last_lap_s=np.array([42.0, np.nan, np.nan, 44.0], np.float32)), 0.004)
    term = np.array([1, 0, 0, 0], bool)
    trunc = np.array([0, 1, 0, 0], bool)
    st.on_step(term, trunc, infos(4, speed_mps=np.array([4.0, 20.0, 1.0, 1.0], np.float32),
                                  term_reason=np.array([TermReason.WALL, TermReason.TIME_LIMIT, 0, 0], np.uint8),
                                  laps=np.array([1, 5, 0, 0], np.uint16)), 0.003)
    m = st.pop()
    assert m["race_episodes"] == 2
    assert m["race_lap_time"] == pytest.approx(43.0) and m["race_lap_time_best"] == pytest.approx(42.0)
    assert m["race_completion_rate"] == 0.5  # truncation = completion (M2 RecordEpisode)
    assert m["race_laps3_rate"] == 0.5 and m["race_laps"] == 3.0
    assert m["race_term_wall"] == 0.5 and m["race_term_timelimit"] == 0.5 and m["race_term_stuck"] == 0.0
    assert m["race_mean_speed"] == pytest.approx(((10 + 10 + 4) / 3 + (10 + 10 + 20) / 3) / 2)
    assert m["step_latency_p50_ms"] == pytest.approx(3.0)
    # the finished agents' accumulators restarted; the pop cleared the window
    assert st.speed_n.tolist() == [0, 0, 3, 3]
    assert math.isnan(st.pop()["race_completion_rate"])


def test_run_config_yaml():
    cfg, run = train_ppo.load_run_config(str(CONFIGS / "ppo_parity.yaml"))
    assert cfg.preset == "parity" and cfg.rollout_total == 20480 and cfg.activation == "silu"
    assert run["eval_seed_base"] == 2000 and run["total_steps"] == 1e7
    cfg_d, _ = train_ppo.load_run_config(str(CONFIGS / "ppo_default.yaml"))
    assert cfg_d.preset == "default" and cfg_d.rollout_total is None


def start_mocks(ports, n, eval_port, crash_first_at=None):
    """K training mocks (n agents) + a 20-agent eval mock. With crash_first_at, mock 0 dies at that STEP and a
    replacement connects to the same port once the trainer relaunches it."""
    mocks = []
    for i, p in enumerate(ports):
        m = MockUnity(p, n, max_episode_decisions=12, crash_at_step=crash_first_at if i == 0 else None)
        th = m.start_thread()
        mocks.append(m)
        if i == 0 and crash_first_at is not None:
            def respawn(th=th, p=p):
                th.join()
                MockUnity(p, n, max_episode_decisions=12).start_thread()
            threading.Thread(target=respawn, daemon=True).start()
    MockUnity(eval_port, 20).start_thread()
    return mocks


def run_train(run_dir, ports, eval_port, *extra, seed=1):
    args = ["--run-dir", str(run_dir), "--exe", "", "--seed", str(seed), "--procs", str(len(ports)), "--agents", "4",
            "--base-port", str(ports[0]), "--eval-port", str(eval_port), "--total-steps", "256",
            "--eval-every", "2", "--ckpt-every", "1", "--milestones", "64", "--no-tensorboard",
            "--config", str(CONFIGS / "ppo_parity.yaml"),
            "--set", "rollout_total=null", "rollout_length=8", "minibatch_size=32", *extra]
    return train_ppo.main(args)


def rows(run_dir, name="metrics.jsonl"):
    return [json.loads(x) for x in (Path(run_dir) / name).read_text(encoding="utf-8").splitlines()]


def mock_ports():
    base, second, ev = free_ports(3)
    return base, ev


def test_train_cli_resume_and_artifacts(tmp_path):
    base, ev = mock_ports()
    ports = [base, base + 25]
    run_dir = tmp_path / "run"
    start_mocks(ports, 4, ev)
    # 2 procs × 4 agents × T 8 = 64 decisions/update, 256 total → 4 updates; stop after 2, then resume
    assert run_train(run_dir, ports, ev, "--max-updates", "2") == 0
    s = json.loads((run_dir / "summary.json").read_text())
    assert s["status"] == "stopped" and s["update"] == 2 and s["global_step"] == 128
    assert (run_dir / "milestone_64.pt").exists() and (run_dir / "best.pt").exists()
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["env_config_hash"] == "90240ee2b1a58b5b" and meta["run"]["eval_seed_base"] == 2000

    start_mocks(ports, 4, ev)
    assert train_ppo.main(["--run-dir", str(run_dir), "--exe", "", "--resume", "--no-tensorboard"]) == 0
    s = json.loads((run_dir / "summary.json").read_text())
    assert s["status"] == "finished" and s["update"] == 4 and s["global_step"] == 256
    hist = rows(run_dir)
    assert [r["global_step"] for r in hist] == [64, 128, 192, 256]
    assert all(r["nan_skipped"] == 0 for r in hist)
    ev_rows = rows(run_dir, "eval.jsonl")
    assert [r["global_step"] for r in ev_rows] == [128, 256] and all(r["seed_base"] == 2000 for r in ev_rows)
    csv_lines = (run_dir / "metrics.csv").read_text().splitlines()
    assert len(csv_lines) == 5
    ck = sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt"))
    assert ck == ["ckpt_128.pt", "ckpt_192.pt", "ckpt_256.pt", "ckpt_64.pt"]
    last = load_checkpoint(run_dir / "checkpoints" / "ckpt_256.pt", "90240ee2b1a58b5b")
    assert last["extra"]["update"] == 4 and last["extra"]["state"]["saved_milestones"] == [64]
    assert last["extra"]["trainer"]["updates"] == 4  # trainer state carried across the resume


def test_train_recovers_from_unity_crash(tmp_path):
    base, ev = mock_ports()
    ports = [base, base + 25]
    start_mocks(ports, 4, ev, crash_first_at=5)
    run_dir = tmp_path / "crash"
    assert run_train(run_dir, ports, ev, "--eval-every", "0") == 0
    s = json.loads((run_dir / "summary.json").read_text())
    assert s["status"] == "finished" and s["restarts"] == 1 and s["global_step"] == 256
    assert len(rows(run_dir)) == 4


def test_first_update_reproducible(tmp_path):
    out = []
    for i in range(2):
        base, ev = mock_ports()
        ports = [base, base + 25]
        start_mocks(ports, 4, ev)
        run_dir = tmp_path / f"r{i}"
        assert run_train(run_dir, ports, ev, "--max-updates", "1", "--eval-every", "0", seed=7) == 0
        out.append(rows(run_dir)[0])
    a, b = ({k: v for k, v in r.items() if k not in TIMING_KEYS} for r in out)
    assert a == b
