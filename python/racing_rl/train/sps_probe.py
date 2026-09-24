"""
SPS probe (M5 SPS procedure, step 1): upper bound of the environment with random actions over an N × K grid.

For each (agents N, processes K) a MultiUnityVecEnv is launched, warmed up, then stepped for --seconds with uniform
random actions. Reported: agent-decisions/s (SPS = vector steps · N · K / s) and the latency of one vector step
(step_async → step_wait over all K processes) p50 / p99. With --policy the actions come from an untrained
ActorCritic (act_full, update_rms=True, torch threads = 2), i.e. the realistic collection loop without the update.

    python -m racing_rl.train.sps_probe --exe ..\\outputs\\builds\\RaceEnv\\RaceEnv.exe --out ..\\outputs\\benchmarks\\eval\\m5_sps_probe.json

Selection rule: the highest-SPS configuration whose p99 step latency is < 50 ms.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from racing_rl.bridge.multi_env import MultiUnityVecEnv

P99_LIMIT_MS = 50.0


def probe(exe: str, n: int, k: int, seconds: float, warmup: int, base_port: int, log_dir: Path,
          policy: bool = False) -> dict:
    t_launch = time.perf_counter()
    envs = MultiUnityVecEnv(exe, num_processes=k, num_agents=n, base_port=base_port, log_dir=log_dir)
    launch_s = time.perf_counter() - t_launch
    try:
        rng = np.random.default_rng(0)
        obs, _ = envs.reset(seed=0)
        ac = None
        if policy:
            import torch

            from racing_rl.rl import ActorCritic, get_preset
            torch.set_num_threads(2)
            ac = ActorCritic.from_config(get_preset("parity"), obs.shape[1], 2)

        def act(o):
            if ac is None:
                return rng.uniform(-1.0, 1.0, (envs.num_envs, 2)).astype(np.float32)
            return ac.act_full(o, update_rms=True)[0]

        for _ in range(warmup):
            obs = envs.step(act(obs))[0]
        lat = []
        steps = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            a = act(obs)
            s0 = time.perf_counter()
            envs.step_async(a)
            obs = envs.step_wait()[0]
            lat.append(time.perf_counter() - s0)
            steps += 1
        wall = time.perf_counter() - t0
    finally:
        envs.close()
    lat_ms = np.asarray(lat) * 1e3
    return {"agents": n, "procs": k, "total_agents": n * k, "policy": policy, "vector_steps": steps,
            "seconds": round(wall, 2), "sps": round(steps * n * k / wall, 1),
            "step_latency_p50_ms": round(float(np.percentile(lat_ms, 50)), 3),
            "step_latency_p99_ms": round(float(np.percentile(lat_ms, 99)), 3),
            "step_latency_max_ms": round(float(lat_ms.max()), 3), "launch_s": round(launch_s, 1)}


def select(rows: list[dict]) -> dict | None:
    ok = [r for r in rows if r["step_latency_p99_ms"] < P99_LIMIT_MS]
    return max(ok, key=lambda r: r["sps"]) if ok else None


def markdown(rows: list[dict]) -> str:
    lines = ["| N (ajan/süreç) | K (süreç) | N·K | SPS (karar/s) | p50 (ms) | p99 (ms) |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['agents']} | {r['procs']} | {r['total_agents']} | {r['sps']:,.0f} | "
                     f"{r['step_latency_p50_ms']:.2f} | {r['step_latency_p99_ms']:.2f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--agents", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--base-port", type=int, default=6305)
    ap.add_argument("--policy", action="store_true", help="actions from an untrained ActorCritic instead of random")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    rows = []
    for n in a.agents:
        for k in a.procs:
            r = probe(a.exe, n, k, a.seconds, a.warmup, a.base_port, out.parent / "unity_logs", a.policy)
            rows.append(r)
            print(json.dumps(r), flush=True)
    best = select(rows)
    rep = {"p99_limit_ms": P99_LIMIT_MS, "rows": rows, "selected": best}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(markdown(rows))
    print("selected:", json.dumps(best))


if __name__ == "__main__":
    main()
