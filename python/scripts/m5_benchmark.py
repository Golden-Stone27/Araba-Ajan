"""
M5 official benchmark (C0.10 over the bridge, test seeds 1000..1019, deterministic μ, same build + evaluator for
both policies). One 20-agent Unity process is reused; every evaluation starts with RESET (fresh cars, C0.17).

    python scripts/m5_benchmark.py --runs ../runs/m5/parity_s1 ../runs/m5/parity_s2 ../runs/m5/parity_s3

Writes
  benchmarks/mlagents_bridge.json  M2 ONNX models (models/mlagents_baseline_s{1,2,3}.onnx)
  benchmarks/custom_ppo.json       per run: best.pt (official, selected on validation seeds 2000..2019, C0.19),
                                   the final checkpoint and the 1M-decision milestone; training statistics
  benchmarks/models/custom_ppo_s{seed}.pt   copies of the evaluated best.pt
  benchmarks/eval/traces/{mlagents,custom}_s{seed}.npz  per-decision traces for compare.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from racing_rl.bridge.evaluate import report  # noqa: E402
from racing_rl.bridge.protocol import FROZEN_ENV_CONFIG_HASH  # noqa: E402
from racing_rl.bridge.vec_env import UnityVecEnv  # noqa: E402
from racing_rl.train.evaluate import DEFAULT_EXE, evaluate  # noqa: E402
from racing_rl.train.train_ppo import latest_checkpoint  # noqa: E402

BENCH = REPO / "benchmarks"
TRACES = BENCH / "eval" / "traces"
EVAL = {"episodes": 20, "seed_base": 1000, "laps": 3, "start": "grid", "deterministic": True}


def summary(per_seed: list[dict]) -> dict:
    meds = [p["flying_lap_median_s"] for p in per_seed if p["flying_lap_median_s"] is not None]
    return {"T_ref_s": statistics.median(meds) if len(meds) == len(per_seed) else None,
            "completion_rate_median": statistics.median(p["completion_rate"] for p in per_seed),
            "completion_rate_min": min(p["completion_rate"] for p in per_seed)}


def show(tag: str, p: dict) -> None:
    print(f"  {tag:<28} completion {p['completion_rate']:.2f}  flying {p['flying_lap_median_s']}  "
          f"best {p['flying_lap_best_s']}  lap1 {p['lap1_median_s']}  steer {p['steer_smoothness']:.4f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="M5 run directories (seed from run.json)")
    ap.add_argument("--exe", default=str(DEFAULT_EXE))
    ap.add_argument("--port", type=int, default=6405)
    ap.add_argument("--skip-mlagents", action="store_true")
    ap.add_argument("--milestone", type=int, default=1_000_000)
    a = ap.parse_args()

    env = UnityVecEnv(a.exe, num_agents=EVAL["episodes"], port=a.port, expected_env_hash=FROZEN_ENV_CONFIG_HASH,
                      log_dir=BENCH / "eval" / "unity_logs")
    try:
        if not a.skip_mlagents:
            print("ML-Agents (M2 ONNX):", flush=True)
            per_seed = []
            for s in (1, 2, 3):
                model = BENCH / "models" / f"mlagents_baseline_s{s}.onnx"
                r = evaluate(env, f"onnx:{model}", s, EVAL["seed_base"], TRACES / f"mlagents_s{s}.npz")["per_seed"]
                r["policy_spec"] = f"onnx:benchmarks/models/{model.name}"
                per_seed.append(r)
                show(f"s{s} {model.name}", r)
            base = json.loads((BENCH / "baseline_mlagents.json").read_text(encoding="utf-8"))
            rep = report(env, per_seed, "mlagents-ppo", {"summary": summary(per_seed), "training": base["training"]})
            rep["eval"] = EVAL
            (BENCH / "mlagents_bridge.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")

        print("Custom PPO:", flush=True)
        best, final, mile, training = [], [], [], []
        for run in map(Path, a.runs):
            meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
            summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            s = int(meta["seed"])
            dst = BENCH / "models" / f"custom_ppo_s{s}.pt"
            shutil.copy2(run / "best.pt", dst)
            r = evaluate(env, f"torch:{dst}", s, EVAL["seed_base"], TRACES / f"custom_s{s}.npz")["per_seed"]
            r.update(policy_spec=f"torch:benchmarks/models/{dst.name}", checkpoint_step=summ["best_step"])
            best.append(r)
            show(f"s{s} best.pt @{summ['best_step']:,}", r)
            ck = latest_checkpoint(run / "checkpoints")
            r = evaluate(env, f"torch:{ck}", s, EVAL["seed_base"])["per_seed"]
            r.update(policy_spec=f"torch:{run.name}/checkpoints/{ck.name}", checkpoint_step=summ["global_step"])
            final.append(r)
            show(f"s{s} final @{summ['global_step']:,}", r)
            mp = run / f"milestone_{a.milestone}.pt"
            if mp.exists():
                r = evaluate(env, f"torch:{mp}", s, EVAL["seed_base"])["per_seed"]
                r.update(policy_spec=f"torch:{run.name}/{mp.name}")
                mile.append(r)
                show(f"s{s} milestone {a.milestone:,}", r)
            training.append({"run_id": run.name, "seed": s, "preset": meta.get("config_file"),
                             "procs": meta["run"]["procs"], "agents": meta["run"]["agents"],
                             "total_env_decisions": summ["global_step"],
                             "decisions_to_first_3lap": summ["decisions_to_first_3lap"],
                             "decisions_to_95pct": summ["decisions_to_95pct"],
                             "wallclock_h": round(summ["wallclock_s"] / 3600.0, 4), "best_step": summ["best_step"],
                             "restarts": summ["restarts"], "git_sha": meta["git_sha"]})
        med = lambda k: statistics.median(t[k] for t in training if t[k] is not None) if any(  # noqa: E731
            t[k] is not None for t in training) else None
        rep = report(env, best, "custom-ppo", {
            "summary": summary(best),
            "final": {"per_seed": final, "summary": summary(final)},
            f"milestone_{a.milestone}": {"per_seed": mile, "summary": summary(mile) if mile else None},
            "training": {"total_env_decisions": sum(t["total_env_decisions"] for t in training),
                         "decisions_to_first_3lap": med("decisions_to_first_3lap"),
                         "decisions_to_95pct": med("decisions_to_95pct"),
                         "wallclock_h": round(sum(t["wallclock_h"] for t in training), 4), "per_seed": training},
            "model_selection": "best.pt on validation seeds 2000..2019 (contracts C0.19); test seeds 1000..1019",
        })
        rep["eval"] = EVAL
        (BENCH / "custom_ppo.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(rep["summary"]), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
