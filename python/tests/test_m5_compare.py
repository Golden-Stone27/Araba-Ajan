"""compare.py on synthetic C0.10 reports and traces: hash guard, tables, plots, flying-lap extraction."""

import json

import numpy as np
import pytest

from racing_rl.train import compare

HASH = "90240ee2b1a58b5b"


def per_seed(seed, lap, completion=1.0):
    return {"seed": seed, "episodes": 20, "completion_rate": completion, "flying_lap_median_s": lap,
            "flying_lap_best_s": lap - 0.05, "lap1_median_s": lap + 3.0, "mean_speed_mps": 26.7,
            "steer_smoothness": 0.02, "term_reasons": {"finished": 20}, "sector_times_s": [13.0, 15.0, lap - 28.0],
            "flying_laps_s": [lap] * 40, "checkpoint_step": 5_000_000}


def rep(policy, laps, h=HASH):
    return {"schema": "race-benchmark/v1", "policy": policy, "evaluator": "bridge", "env_config_hash": h,
            "obs_layout_hash": "b40ca79bdba1c2c2", "unity": "6000.4.6f1", "build_id": "x",
            "eval": {"episodes": 20, "seed_base": 1000, "laps": 3, "start": "grid", "deterministic": True},
            "per_seed": [per_seed(i + 1, t) for i, t in enumerate(laps)],
            "training": {"total_env_decisions": 3e7, "decisions_to_first_3lap": 400000, "decisions_to_95pct": 700000,
                         "wallclock_h": 1.0, "per_seed": []}}


def trace(n_steps=1400, n=20):
    t = np.arange(n_steps)
    prog = (t % 440) / 440.0
    laps = np.repeat((t // 440)[:, None], n, 1)
    tr = {"pos_x": np.repeat(np.cos(prog * 2 * np.pi)[:, None], n, 1) * 100,
          "pos_z": np.repeat(np.sin(prog * 2 * np.pi)[:, None], n, 1) * 100,
          "speed_mps": np.full((n_steps, n), 27.0), "progress": np.repeat(prog[:, None], n, 1),
          "laps": laps, "e_lat": np.zeros((n_steps, n)), "steer": np.zeros((n_steps, n)),
          "throttle": np.ones((n_steps, n)), "done": np.zeros((n_steps, n), bool)}
    tr["done"][1330, :] = True
    return tr


def test_hash_mismatch_refused(tmp_path):
    with pytest.raises(compare.HashMismatchError):
        compare.build(rep("mlagents-ppo", [41.2] * 3), rep("custom-ppo", [41.0] * 3, h="deadbeefdeadbeef"), [],
                      tmp_path, tmp_path / "plots", None, "plots")


def test_flying_lap_extraction():
    lap = compare.flying_lap(trace(), agent=0, lap=2)
    assert lap is not None and 400 < len(lap["progress"]) < 440
    assert lap["progress"].min() < 0.01 and lap["progress"].max() < 0.999
    # a stale ~1.0 progress sample right after the line crossing is dropped (no 100 % → 0 % wrap segment)
    tr = trace()
    tr["progress"][441, :] = 0.995
    lap = compare.flying_lap(tr, agent=0, lap=2)
    assert np.all(np.diff(lap["progress"]) > 0)


def test_build_report_and_plots(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    for kind in ("mlagents", "custom"):
        for s in (1, 2, 3):
            np.savez_compressed(traces / f"{kind}_s{s}.npz", **trace(), dt_dec=0.1)
    run = tmp_path / "parity_s1"
    run.mkdir()
    (run / "metrics.jsonl").write_text("".join(json.dumps({"global_step": 20480 * i, "ep_return_mean": float(i),
                                                           "race_lap_time": 50.0 - i * 0.1,
                                                           "race_completion_rate": min(1.0, i / 50)}) + "\n"
                                               for i in range(1, 100)))
    (run / "eval.jsonl").write_text("".join(json.dumps({"global_step": 512000 * i, "flying_lap_median_s": 45 - i,
                                                        "completion_rate": 1.0}) + "\n" for i in range(1, 4)))
    ml = rep("mlagents-ppo", [41.28, 41.22, 41.08])
    ml["training"]["per_seed"] = [{"curve": [{"step": 100000 * i, "reward": i, "lap_time_s": 50 - i,
                                              "completion_rate": i / 10} for i in range(1, 11)],
                                   "total_env_decisions": 1e7, "seed": 1}]
    cu = rep("custom-ppo", [41.0, 41.3, 41.1])
    cu["final"] = {"per_seed": cu["per_seed"], "summary": {"T_ref_s": 41.1, "completion_rate_median": 1.0}}
    md = compare.build(ml, cu, [run], traces, tmp_path / "plots", "Notlar burada.", "plots")
    assert "✅" in md and "41.63" in md and "Notlar burada." in md
    assert "| Flying lap medyanı (T, s) | 41.22 | **41.10** | -0.29% ✅ | 41.10 | -0.29% ✅ |" in md
    assert "Özel son | Δ% (son) |" in md
    for f in ("learning_curves.png", "lap_time_box.png", "racing_line.png", "speed_steer_profile.png",
              "sector_deltas.png"):
        assert (tmp_path / "plots" / f).stat().st_size > 10_000
