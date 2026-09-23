"""
C0.10 benchmark over the bridge (evaluator "bridge"): N = 20 agents, EvalGrid (seed 1000 + i), 3 laps, deterministic
policy, first episode per agent. Mirrors Racing.Core.BenchmarkRecorder; differences are sampling-only:
mean speed and sector splits use per-decision (0.1 s) samples instead of per-physics-step (0.02 s) ones.

    python -m racing_rl.bridge.evaluate --exe ..\\Builds\\RaceEnv\\RaceEnv.exe \\
        --model ..\\benchmarks\\models\\mlagents_baseline_s1.onnx --train-seed 1 --out ..\\benchmarks\\eval\\bridge_s1.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .policies import OnnxPolicy, Policy
from .protocol import TERM_REASON_KEYS, StartMode, TermReason
from .vec_env import UnityVecEnv

EPISODES = 20
SEED_BASE = 1000
LAPS = 3
SCHEMA = "race-benchmark/v1"


@dataclass
class _Episode:
    done: bool = False
    reason: int = 0
    laps: list = field(default_factory=list)
    sectors: list = field(default_factory=list)  # per lap: [s1, s2, s3] or None
    marks: list = field(default_factory=list)  # absolute split times within the current lap
    lap_start: float = 0.0
    prev_t: float = 0.0
    prev_p: float = 0.0
    speed_sum: float = 0.0
    speed_n: int = 0
    steer_sum: float = 0.0
    prev_steer: float = 0.0
    decisions: int = 0


def _nan_to_none(v):
    return None if v is None or not np.isfinite(v) else float(v)


def run_eval(env: UnityVecEnv, policy: Policy, episodes: int = EPISODES, seed_base: int = SEED_BASE, laps: int = LAPS,
             max_decisions: int = 4000) -> list[_Episode]:
    if env.num_envs != episodes:
        raise ValueError(f"env has {env.num_envs} agents, eval needs {episodes}")
    dt_dec = env.hello["fixed_dt"] * env.hello["decision_period"]
    obs, info = env.reset(seed=seed_base, options={"start_mode": int(StartMode.EVAL_GRID), "max_laps": laps})
    eps = [_Episode() for _ in range(episodes)]
    for e, p in zip(eps, info["progress"]):
        e.prev_p = float(p)
    for _ in range(max_decisions):
        if all(e.done for e in eps):
            break
        actions = np.asarray(policy(obs), np.float32)
        obs, _, term, trunc, info = env.step(actions)
        applied = np.clip(actions, -1.0, 1.0)
        for i, e in enumerate(eps):
            if e.done:
                continue
            steer = float(applied[i, 0])
            e.steer_sum += abs(steer - e.prev_steer)
            e.prev_steer = steer
            e.speed_sum += float(info["speed_mps"][i])
            e.speed_n += 1
            t = int(info["ep_decisions"][i]) * dt_dec
            p = float(info["progress"][i])
            lap_done = bool(info["lap_completed"][i])
            # Sector splits at L/3 and 2L/3: interpolate between decision samples (progress wraps at the line).
            if not lap_done and p >= e.prev_p:
                for b in (1 / 3, 2 / 3):
                    if len(e.marks) < 2 and e.prev_p < b <= p and p - e.prev_p < 0.2:
                        frac = (b - e.prev_p) / (p - e.prev_p)
                        e.marks.append(e.prev_t + frac * (t - e.prev_t))
            if lap_done:
                lt = float(info["last_lap_s"][i])
                e.laps.append(lt)
                if len(e.marks) == 2:
                    s1 = e.marks[0] - e.lap_start
                    s2 = e.marks[1] - e.marks[0]
                    e.sectors.append([s1, s2, lt - s1 - s2])
                else:
                    e.sectors.append(None)
                e.lap_start += lt
                e.marks = []
            e.prev_t, e.prev_p = t, p
            if term[i] or trunc[i]:
                e.done = True
                e.reason = int(info["term_reason"][i])
                e.decisions = int(info["ep_decisions"][i])
    return eps


def per_seed_report(eps: list[_Episode], train_seed: int, laps: int = LAPS) -> dict:
    flying, lap1, s1, s2, s3 = [], [], [], [], []
    reasons: dict[str, int] = {}
    success = 0
    speed_sum = speed_n = 0.0
    steer_sum = decisions = 0.0
    for e in eps:
        key = TERM_REASON_KEYS[TermReason(e.reason if e.done else 0)]
        reasons[key] = reasons.get(key, 0) + 1
        speed_sum += e.speed_sum
        speed_n += e.speed_n
        steer_sum += e.steer_sum
        decisions += e.decisions
        if not (e.done and e.reason == TermReason.FINISHED and len(e.laps) >= laps):
            continue
        success += 1
        lap1.append(e.laps[0])
        for lap, sec in zip(e.laps[1:], e.sectors[1:]):
            flying.append(lap)
            if sec is not None:
                s1.append(sec[0])
                s2.append(sec[1])
                s3.append(sec[2])
    med = lambda v: float(np.median(v)) if v else None  # noqa: E731
    return {
        "seed": train_seed,
        "episodes": len(eps),
        "completion_rate": success / len(eps) if eps else 0.0,
        "flying_lap_median_s": med(flying),
        "flying_lap_best_s": float(min(flying)) if flying else None,
        "lap1_median_s": med(lap1),
        "mean_speed_mps": speed_sum / speed_n if speed_n else 0.0,
        "steer_smoothness": steer_sum / decisions if decisions else 0.0,
        "term_reasons": dict(sorted(reasons.items())),
        "sector_times_s": [med(s1), med(s2), med(s3)],
        "flying_laps_s": [round(x, 4) for x in flying],
    }


def report(env: UnityVecEnv, per_seed: list[dict], policy_name: str, extra: dict | None = None) -> dict:
    out = {
        "schema": SCHEMA,
        "policy": policy_name,
        "evaluator": "bridge",
        "env_config_hash": env.hello["env_config_hash"],
        "obs_layout_hash": env.hello["obs_layout_hash"],
        "unity": env.hello["unity"],
        "build_id": env.hello["build_id"],
        "train_seeds": [p["seed"] for p in per_seed],
        "eval": {"episodes": EPISODES, "seed_base": SEED_BASE, "laps": LAPS, "start": "grid", "deterministic": True},
        "per_seed": per_seed,
    }
    if extra:
        out.update(extra)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=6005)
    ap.add_argument("--policy-name", default="mlagents-ppo")
    a = ap.parse_args()
    policy = OnnxPolicy(a.model)
    env = UnityVecEnv(a.exe, num_agents=EPISODES, port=a.port, log_dir=Path(a.out).parent / "unity_logs")
    try:
        t0 = time.perf_counter()
        eps = run_eval(env, policy)
        rep = report(env, [per_seed_report(eps, a.train_seed)], a.policy_name,
                     {"model": Path(a.model).name, "wallclock_s": round(time.perf_counter() - t0, 2)})
    finally:
        env.close()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rep["per_seed"][0], indent=2))


if __name__ == "__main__":
    main()
