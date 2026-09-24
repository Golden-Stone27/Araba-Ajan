"""Builds outputs/benchmarks/baseline_mlagents.json (C0.10 schema) from per-seed eval files + TensorBoard curves.

Usage (from the ASCII junction D:\\RaceAgent, Python = python/.venv-mla):
  python python/baselines/mlagents/make_baseline_report.py --seed 1:outputs/benchmarks/eval/mlagents_s1.json:baseline_s1:RaceCar-999981 [--seed 2:...]

Each --seed is  <train_seed>:<eval json>:<run id>:<evaluated checkpoint name>.
Training summary per seed: total env decisions at the evaluated checkpoint, first summary step with
Race/Laps3Rate > 0 (decisions_to_first_3lap), first step with Race/CompletionRate >= 0.95
(decisions_to_95pct), wall clock up to the checkpoint and a down-sampled curve.
"""
import argparse
import glob
import json
import re
import statistics
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # racing_rl.paths only; .venv-mla has no racing_rl

from racing_rl import paths  # noqa: E402

RESULTS = paths.RUNS / "mlagents"
CURVE_TAGS = {
    "reward": "Environment/Cumulative Reward",
    "episode_length": "Environment/Episode Length",
    "completion_rate": "Race/CompletionRate",
    "stuck_rate": "Race/TermReason/Stuck",
    "wall_rate": "Race/TermReason/Wall",
    "laps": "Race/Laps",
    "lap_time_s": "Race/LapTime",
    "mean_speed_mps": "Race/MeanSpeed",
    "entropy": "Policy/Entropy",
}


def load_scalars(run_id):
    files = sorted(glob.glob(str(RESULTS / run_id / "RaceCar" / "events.out.tfevents.*")))
    out = {}
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            out.setdefault(tag, {}).update({e.step: (e.value, e.wall_time) for e in ea.Scalars(tag)})
    return out


def first_step(series, pred, upto):
    for step in sorted(series):
        if step > upto:
            break
        if pred(series[step][0]):
            return step
    return None


def training_summary(run_id, ckpt_step):
    sc = load_scalars(run_id)
    reward = sc["Environment/Cumulative Reward"]
    steps = [s for s in sorted(reward) if s <= ckpt_step]
    t0 = min(v[1] for v in reward.values())
    # wall clock at the last summary before the checkpoint (+ one summary period of slack is ignored)
    wall_h = (reward[steps[-1]][1] - t0) / 3600.0 if steps else 0.0
    # down-sampled curve: every 100k decisions up to the checkpoint
    curve = []
    for s in steps:
        if s % 100000 and s != steps[-1]:
            continue
        row = {"step": s}
        for key, tag in CURVE_TAGS.items():
            v = sc.get(tag, {}).get(s)
            row[key] = round(v[0], 4) if v else None
        curve.append(row)
    return {
        "run_id": run_id,
        "total_env_decisions": ckpt_step,
        "decisions_to_first_3lap": first_step(sc.get("Race/Laps3Rate", {}), lambda v: v > 0, ckpt_step),
        "decisions_to_95pct": first_step(sc.get("Race/CompletionRate", {}), lambda v: v >= 0.95, ckpt_step),
        "wallclock_h": round(wall_h, 4),
        "curve": curve,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="append", required=True)
    ap.add_argument("--out", default=str(paths.BENCHMARKS / "baseline_mlagents.json"))
    args = ap.parse_args()

    per_seed, training, seeds, header = [], [], [], None
    for spec in args.seed:
        seed_s, eval_path, run_id, ckpt = spec.split(":")
        rep = json.loads(Path(eval_path).read_text(encoding="utf-8"))
        if header is None:
            header = rep
        elif rep["env_config_hash"] != header["env_config_hash"]:
            raise SystemExit(f"env_config_hash mismatch: {eval_path}")
        entry = dict(rep["per_seed"][0])
        entry["seed"] = int(seed_s)
        entry["model"] = paths.rel(paths.MODELS / f"mlagents_baseline_s{seed_s}.onnx")
        entry["checkpoint"] = ckpt
        entry["build_id"] = rep["build_id"]
        per_seed.append(entry)
        seeds.append(int(seed_s))
        ckpt_step = int(re.search(r"(\d+)$", ckpt).group(1))
        t = training_summary(run_id, ckpt_step)
        t["seed"] = int(seed_s)
        training.append(t)

    ok = [p["flying_lap_median_s"] for p in per_seed if p["flying_lap_median_s"] is not None]
    report = {
        "schema": "race-benchmark/v1",
        "policy": "mlagents-ppo",
        "evaluator": header["evaluator"],
        "env_config_hash": header["env_config_hash"],
        "obs_layout_hash": header["obs_layout_hash"],
        "unity": header["unity"],
        "build_id": "RaceEnv_MLA (checkpoint per seed: per_seed[].build_id)",
        "train_seeds": seeds,
        "eval": header["eval"],
        "per_seed": per_seed,
        "summary": {
            "T_ref_s": statistics.median(ok) if ok else None,
            "completion_rate_median": statistics.median(p["completion_rate"] for p in per_seed),
        },
        "training": {
            "total_env_decisions": sum(t["total_env_decisions"] for t in training),
            "decisions_to_first_3lap": statistics.median(
                [t["decisions_to_first_3lap"] for t in training if t["decisions_to_first_3lap"] is not None] or [None]),
            "decisions_to_95pct": statistics.median(
                [t["decisions_to_95pct"] for t in training if t["decisions_to_95pct"] is not None] or [None]),
            "wallclock_h": round(sum(t["wallclock_h"] for t in training), 4),
            "per_seed": training,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("summary", "train_seeds")}, indent=2))
    print({k: v for k, v in report["training"].items() if k != "per_seed"})


if __name__ == "__main__":
    main()
