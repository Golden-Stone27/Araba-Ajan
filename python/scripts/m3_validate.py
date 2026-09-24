"""
M3 DoD validation against the bridge player.
Writes outputs/benchmarks/eval/m3_bridge_validation.json and outputs/benchmarks/eval/bridge_s{1,2,3}.json.

    python scripts/m3_validate.py                      # everything
    python scripts/m3_validate.py --only parity determinism
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing_rl import paths  # noqa: E402
from racing_rl.bridge import OnnxPolicy, RacingEnv, UnityVecEnv  # noqa: E402
from racing_rl.bridge.evaluate import per_seed_report, report, run_eval  # noqa: E402

EXE = paths.BRIDGE_EXE
EVAL_DIR = paths.EVAL
RUNS = paths.RUNS / "m3"
BASELINE = paths.BENCHMARKS / "baseline_mlagents.json"


def dod2_check_env() -> dict:
    from gymnasium.utils.env_checker import check_env

    env = RacingEnv(EXE, num_agents=1, log_dir=RUNS / "unity_logs")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_env(env, skip_render_check=True)
    env.close()
    return {"passed": True, "warnings": sorted({str(x.message)[:160] for x in w})}


def _rollout(seed: int, decisions: int, n: int) -> tuple[dict, dict]:
    env = UnityVecEnv(EXE, num_agents=n, log_dir=RUNS / "unity_logs")
    actions = np.random.default_rng(7).uniform(-1, 1, (decisions, n, 2)).astype(np.float32)
    h = {k: hashlib.sha256() for k in ("obs", "reward", "terminated", "truncated", "final_obs", "info")}
    arrays = {k: [] for k in ("obs", "reward", "terminated", "truncated")}
    dones = 0
    try:
        obs, info = env.reset(seed=seed)
        h["obs"].update(obs.tobytes())
        arrays["obs"].append(obs)
        for t in range(decisions):
            obs, rew, term, trunc, info = env.step(actions[t])
            for k, v in (("obs", obs), ("reward", rew), ("terminated", term), ("truncated", trunc)):
                h[k].update(v.tobytes())
                arrays[k].append(v)
            h["final_obs"].update(info["final_obs"].tobytes())
            h["info"].update(np.stack([info[f] for f in ("laps", "ep_decisions", "ep_return", "pos_x", "pos_z")]).tobytes())
            dones += int((term | trunc).sum())
    finally:
        env.close()
    return {k: v.hexdigest()[:16] for k, v in h.items()} | {"dones": dones}, {k: np.stack(v) for k, v in arrays.items()}


def dod4_determinism(decisions: int = 5000, n: int = 16, seed: int = 123) -> dict:
    a_h, a = _rollout(seed, decisions, n)
    b_h, b = _rollout(seed, decisions, n)
    equal = {k: bool(np.array_equal(a[k], b[k])) for k in a}
    first_diff = None
    if not all(equal.values()):
        diff = np.flatnonzero(np.any(a["obs"] != b["obs"], axis=(1, 2)))
        first_diff = int(diff[0]) if diff.size else None
    return {"decisions": decisions, "num_agents": n, "seed": seed, "action_rng": "np.random.default_rng(7)",
            "auto_resets": a_h["dones"], "array_equal": equal, "stream_sha256_run1": a_h, "stream_sha256_run2": b_h,
            "passed": all(equal.values()) and a_h == b_h, "first_divergent_decision": first_diff}


def dod3_5_7_stress(steps: int = 100_000, n: int = 16) -> dict:
    """Lossless (DoD 3), reward consistency (DoD 5) and performance (DoD 7) from one 100k-STEP batchmode run."""
    timing = RUNS / "stress_timing.bin"
    env = UnityVecEnv(EXE, num_agents=n, log_dir=RUNS / "unity_logs", record_rtt=True, timing_log=timing)
    rng = np.random.default_rng(11)
    ep_sum = np.zeros(n)
    diffs = []
    episodes = 0
    t0 = time.perf_counter()
    err = None
    try:
        env.reset(seed=2024)
        for _ in range(steps):
            _, rew, term, trunc, info = env.step(rng.uniform(-1, 1, (n, 2)).astype(np.float32))
            ep_sum += rew
            for i in np.flatnonzero(term | trunc):
                diffs.append(abs(ep_sum[i] - float(info["ep_return"][i])))
                ep_sum[i] = 0.0
                episodes += 1
    except Exception as e:  # noqa: BLE001 - recorded as a DoD failure
        err = repr(e)
    wall = time.perf_counter() - t0
    sent, recv = env.requests_sent, env.replies_received
    nonfinite = env.nonfinite_obs
    rtt_seq = np.array(env.rtt_seq, np.int64)
    rtt_us = np.array(env.rtt_ns, np.float64) / 1e3
    env.close()
    time.sleep(0.5)
    stats = json.loads((Path(str(timing) + ".json")).read_text(encoding="utf-8"))
    rec = np.fromfile(timing, dtype=np.dtype([("seq", "<u4"), ("type", "<u2"), ("pad", "<u2"), ("unity_us", "<f4")]))
    step_rec = rec[rec["type"] == 0x0011]
    unity_by_seq = dict(zip(step_rec["seq"].tolist(), step_rec["unity_us"].tolist()))
    mask = np.array([s in unity_by_seq for s in rtt_seq.tolist()])
    rtt_step = rtt_us[mask]
    unity_step = np.array([unity_by_seq[s] for s in rtt_seq[mask].tolist()])
    overhead = rtt_step - unity_step
    pct = lambda v, q: float(np.percentile(v, q))  # noqa: E731
    diffs = np.array(diffs)
    return {
        "lossless": {
            "steps": steps, "num_agents": n, "agent_decisions": steps * n, "error": err,
            "python_requests_sent": sent, "python_replies_received": recv,
            "unity_messages_received": stats["messages_received"], "unity_messages_sent": stats["messages_sent"],
            "unity_steps": stats["steps"], "passed": err is None and sent == recv and stats["steps"] == steps,
        },
        "reward_consistency": {
            "episodes": episodes, "max_abs_diff": float(diffs.max()) if diffs.size else None,
            "mean_abs_diff": float(diffs.mean()) if diffs.size else None,
            "passed": bool(diffs.size and diffs.max() <= 1e-4),
        },
        "nonfinite_obs_agents": nonfinite,
        "zero_gc": {"gc_steady_bytes": stats["gc_steady_bytes"], "gc_steady_steps": stats["gc_steady_steps"],
                    "passed": stats["gc_steady_bytes"] == 0},
        "performance": {
            "wallclock_s": round(wall, 2),
            "agent_decisions_per_s": round(steps * n / wall, 1),
            "steps_per_s": round(steps / wall, 1),
            "rtt_us": {"p50": pct(rtt_step, 50), "p99": pct(rtt_step, 99), "mean": float(rtt_step.mean())},
            "unity_step_us": {"p50": pct(unity_step, 50), "p99": pct(unity_step, 99), "mean": float(unity_step.mean())},
            "bridge_overhead_us": {"p50": pct(overhead, 50), "p90": pct(overhead, 90), "p99": pct(overhead, 99),
                                   "max": float(overhead.max())},
            "paired_samples": int(mask.sum()),
            "passed": steps * n / wall >= 2000 and pct(overhead, 50) < 300 and pct(overhead, 99) < 2000,
        },
    }


def dod6_parity(seeds=(1, 2, 3)) -> dict:
    base = {p["seed"]: p for p in json.loads(BASELINE.read_text(encoding="utf-8"))["per_seed"]}
    out = {}
    for s in seeds:
        model = paths.MODELS / f"mlagents_baseline_s{s}.onnx"
        policy = OnnxPolicy(model)
        env = UnityVecEnv(EXE, num_agents=20, log_dir=RUNS / "unity_logs")
        try:
            t0 = time.perf_counter()
            eps = run_eval(env, policy)
            ps = per_seed_report(eps, s)
            rep = report(env, [ps], "mlagents-ppo", {"model": model.name, "onnx_io": {"inputs": policy.input_names,
                         "output": policy.output}, "wallclock_s": round(time.perf_counter() - t0, 2)})
        finally:
            env.close()
        (EVAL_DIR / f"bridge_s{s}.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        m = base[s]
        d_cr = ps["completion_rate"] - m["completion_rate"]
        d_fl = (ps["flying_lap_median_s"] - m["flying_lap_median_s"]) / m["flying_lap_median_s"]
        fb, fm = np.array(ps["flying_laps_s"]), np.array(m.get("flying_laps_s", []))
        same = fb.shape == fm.shape and bool(np.all(np.abs(fb - fm) < 1e-3))
        out[f"s{s}"] = {
            "bridge": {k: ps[k] for k in ("completion_rate", "flying_lap_median_s", "flying_lap_best_s", "lap1_median_s",
                                          "mean_speed_mps", "steer_smoothness", "term_reasons", "sector_times_s")},
            "mlagents_inproc": {k: m[k] for k in ("completion_rate", "flying_lap_median_s", "flying_lap_best_s",
                                                  "lap1_median_s", "mean_speed_mps", "steer_smoothness", "term_reasons",
                                                  "sector_times_s")},
            "delta_completion_pts": round(100 * d_cr, 3),
            "delta_flying_median_pct": round(100 * d_fl, 4),
            "flying_laps_identical": same,
            "passed": abs(d_cr) <= 0.05 and abs(d_fl) <= 0.01,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, choices=["check_env", "determinism", "stress", "parity"])
    ap.add_argument("--stress-steps", type=int, default=100_000)
    ap.add_argument("--out", default=str(EVAL_DIR / "m3_bridge_validation.json"))
    a = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(a.out)
    res = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() and a.only else {}
    res["meta"] = {"exe": paths.rel(EXE), "python": platform.python_version(), "numpy": np.__version__,
                   "date": time.strftime("%Y-%m-%d %H:%M")}
    todo = a.only or ["check_env", "determinism", "stress", "parity"]
    for name in todo:
        t0 = time.perf_counter()
        print(f"== {name}", flush=True)
        if name == "check_env":
            res["dod2_check_env"] = dod2_check_env()
        elif name == "determinism":
            res["dod4_determinism"] = dod4_determinism()
        elif name == "stress":
            r = dod3_5_7_stress(a.stress_steps)
            res["dod3_lossless"] = r["lossless"]
            res["dod5_reward_consistency"] = r["reward_consistency"]
            res["dod7_performance"] = r["performance"]
            res["zero_gc"] = r["zero_gc"]
            res["nonfinite_obs_agents"] = r["nonfinite_obs_agents"]
        elif name == "parity":
            res["dod6_parity"] = dod6_parity()
        print(f"   done in {time.perf_counter() - t0:.1f}s", flush=True)
        out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
