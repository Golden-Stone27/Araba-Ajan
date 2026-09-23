"""
C0.10 evaluation over the bridge for both policy kinds (the official M5 comparison uses this for ML-Agents and
custom PPO alike: same build, same evaluator, same seeds).

    python -m racing_rl.train.evaluate --policy torch:../runs/m5/parity_s1/best.pt --train-seed 1 \\
        --out ../benchmarks/eval/m5_custom_s1.json --trace ../benchmarks/eval/traces/custom_s1.npz
    python -m racing_rl.train.evaluate --policy onnx:../benchmarks/models/mlagents_baseline_s1.onnx --train-seed 1 ...

--policy torch:<ckpt> | onnx:<path>. Defaults: 20 episodes, EvalGrid, seed base 1000 (test set; training-time model
selection uses 2000, contracts C0.19), 3 laps, deterministic μ. --trace stores per-decision info of every agent
(pos_x, pos_z, speed_mps, progress, laps, ep_decisions, e_lat in m, steer / throttle actions, done) for the
racing-line, speed and sector plots in compare.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from racing_rl.bridge.evaluate import EPISODES, LAPS, SEED_BASE, per_seed_report, report, run_eval
from racing_rl.bridge.policies import OnnxPolicy
from racing_rl.bridge.protocol import FROZEN_ENV_CONFIG_HASH, FROZEN_OBS_LAYOUT_HASH
from racing_rl.bridge.vec_env import UnityVecEnv

REPO = Path(__file__).resolve().parents[3]
DEFAULT_EXE = REPO / "Builds" / "RaceEnv" / "RaceEnv.exe"
TRACE_FIELDS = ("pos_x", "pos_z", "speed_mps", "progress", "laps", "ep_decisions", "lap_completed", "last_lap_s",
                "term_reason")
E_LAT_IDX, E_LAT_SCALE = 22, 6.0


class RecordingEnv:
    """Proxy for run_eval (needs num_envs, hello, reset, step) that records every reply's infos and actions."""

    def __init__(self, env: UnityVecEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.hello = env.hello
        self.rows: dict[str, list] = {k: [] for k in (*TRACE_FIELDS, "e_lat", "steer", "throttle", "done")}

    def reset(self, **kw):
        return self.env.reset(**kw)

    def step(self, actions):
        obs, rew, term, trunc, info = self.env.step(actions)
        a = np.clip(np.asarray(actions, np.float32), -1.0, 1.0)
        for k in TRACE_FIELDS:
            self.rows[k].append(np.asarray(info[k]).copy())
        self.rows["e_lat"].append(np.asarray(obs)[:, E_LAT_IDX] * E_LAT_SCALE)  # C0.6: e_lat / 6 (clipped ±1.5)
        self.rows["steer"].append(a[:, 0].copy())
        self.rows["throttle"].append(a[:, 1].copy())
        self.rows["done"].append(np.asarray(term | trunc, bool))
        return obs, rew, term, trunc, info

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.stack(v) for k, v in self.rows.items() if v}


def load_policy(spec: str):
    kind, _, path = spec.partition(":")
    if kind == "onnx":
        return OnnxPolicy(path), "mlagents-ppo", Path(path).name
    if kind == "torch":
        import torch

        from racing_rl.rl import TorchPolicy
        torch.set_num_threads(1)
        pol = TorchPolicy.from_checkpoint(path, deterministic=True, expected_env_hash=FROZEN_ENV_CONFIG_HASH,
                                          expected_obs_hash=FROZEN_OBS_LAYOUT_HASH)
        return pol, "custom-ppo", Path(path).name
    raise ValueError(f"--policy must be torch:<ckpt> or onnx:<path>, got {spec!r}")


def evaluate(env: UnityVecEnv, spec: str, train_seed: int, seed_base: int = SEED_BASE,
             trace: str | Path | None = None) -> dict:
    policy, name, model = load_policy(spec)
    rec = RecordingEnv(env)
    t0 = time.perf_counter()
    eps = run_eval(rec, policy, episodes=env.num_envs, seed_base=seed_base)
    ps = per_seed_report(eps, train_seed)
    ps.update(model=model, policy_spec=spec, seed_base=seed_base, wallclock_s=round(time.perf_counter() - t0, 2))
    if trace is not None:
        Path(trace).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace, **rec.arrays(), dt_dec=env.hello["fixed_dt"] * env.hello["decision_period"])
        ps["trace"] = str(Path(trace).name)
    return {"policy": name, "per_seed": ps}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--train-seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace")
    ap.add_argument("--exe", default=str(DEFAULT_EXE))
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--port", type=int, default=6405)
    a = ap.parse_args()
    out = Path(a.out)
    env = UnityVecEnv(a.exe, num_agents=a.episodes, port=a.port, log_dir=out.parent / "unity_logs")
    try:
        res = evaluate(env, a.policy, a.train_seed, a.seed_base, a.trace)
        rep = report(env, [res["per_seed"]], res["policy"])
    finally:
        env.close()
    rep["eval"] = {"episodes": a.episodes, "seed_base": a.seed_base, "laps": LAPS, "start": "grid",
                   "deterministic": True}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    ps = {k: v for k, v in rep["per_seed"][0].items() if k != "flying_laps_s"}
    print(json.dumps(ps, indent=2))


if __name__ == "__main__":
    main()
