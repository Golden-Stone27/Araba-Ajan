"""
Training telemetry (M5): per-step race statistics from the bridge infos, TensorBoard + CSV/JSONL writers.

Tag names follow ML-Agents so the curves overlay (x axis = environment decisions, like ML-Agents "Step"):
Environment/*, Losses/*, Policy/*; plus Race/*, Train/*, Perf/*, Eval/*. The Race/* episode statistics mirror
MlaSimulationDriver.RecordEpisode (M2): CompletionRate = episode ended by truncation, Laps3Rate = laps ≥ 3,
TermReason/<Name> = share of finished episodes per reason, LapTime = every completed lap.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from racing_rl.bridge.protocol import TermReason

# C# enum names (Race/TermReason/<Name> in M2's TensorBoard).
REASON_NAMES = {TermReason.WALL: "Wall", TermReason.WRONG_WAY: "WrongWay", TermReason.STUCK: "Stuck",
                TermReason.FLIP: "Flip", TermReason.OUT_OF_BOUNDS: "OutOfBounds", TermReason.TIME_LIMIT: "TimeLimit",
                TermReason.PHYSICS_ERROR: "PhysicsError", TermReason.FINISHED: "Finished"}

# metric key (PPOTrainer / train loop / RaceStats) → TensorBoard tag
TAGS = {
    "ep_return_mean": "Environment/Cumulative Reward",
    "ep_length_mean": "Environment/Episode Length",
    "policy_loss": "Losses/Policy Loss",
    "value_loss": "Losses/Value Loss",
    "entropy_per_dim": "Policy/Entropy",  # ML-Agents reports the mean over action dims
    "entropy": "Train/EntropySum",
    "lr": "Policy/Learning Rate",
    "epsilon": "Policy/Epsilon",
    "beta": "Policy/Beta",
    "approx_kl": "Train/ApproxKL",
    "clip_frac": "Train/ClipFrac",
    "explained_var": "Train/ExplainedVar",
    "grad_norm": "Train/GradNorm",
    "sigma_steer": "Train/Sigma_steer",
    "sigma_thr": "Train/Sigma_thr",
    "lr_mult": "Train/LrMult",
    "nan_skipped": "Train/NanSkipped",
    "sps": "Perf/SPS",
    "collect_sps": "Perf/CollectSPS",
    "time_update_s": "Perf/UpdateSeconds",
    "step_latency_p50_ms": "Perf/StepLatencyP50",
    "step_latency_p99_ms": "Perf/StepLatencyP99",
    "race_lap_time": "Race/LapTime",
    "race_lap_time_best": "Race/LapTimeBest",
    "race_completion_rate": "Race/CompletionRate",
    "race_laps3_rate": "Race/Laps3Rate",
    "race_laps": "Race/Laps",
    "race_mean_speed": "Race/MeanSpeed",
    "race_episodes": "Race/Episodes",
}
TAGS.update({f"race_term_{n.lower()}": f"Race/TermReason/{n}" for n in REASON_NAMES.values()})

EVAL_TAGS = {
    "completion_rate": "Eval/CompletionRate",
    "flying_lap_median_s": "Eval/FlyingLapMedian",
    "flying_lap_best_s": "Eval/FlyingLapBest",
    "lap1_median_s": "Eval/Lap1Median",
    "mean_speed_mps": "Eval/MeanSpeed",
    "steer_smoothness": "Eval/SteerSmoothness",
}


class RaceStats:
    """Vectorised per-step aggregation of RACE_INFO_V1 (no per-agent Python loop)."""

    def __init__(self, num_envs: int):
        self.speed_sum = np.zeros(num_envs, np.float64)
        self.speed_n = np.zeros(num_envs, np.int64)
        self.clear()

    def clear(self) -> None:
        self.lap_times: list[np.ndarray] = []
        self.reasons: list[np.ndarray] = []
        self.laps: list[np.ndarray] = []
        self.truncs: list[np.ndarray] = []
        self.speeds: list[np.ndarray] = []
        self.latencies: list[float] = []

    def discard_running(self) -> None:
        """A rollout was thrown away (Unity restart): running per-agent accumulators restart too."""
        self.speed_sum[:] = 0.0
        self.speed_n[:] = 0

    def on_step(self, term: np.ndarray, trunc: np.ndarray, infos: dict, latency_s: float) -> None:
        self.latencies.append(latency_s)
        self.speed_sum += infos["speed_mps"]
        self.speed_n += 1
        lap = infos["lap_completed"].astype(bool)
        if lap.any():
            self.lap_times.append(infos["last_lap_s"][lap].astype(np.float64))
        done = term | trunc
        if done.any():
            self.reasons.append(infos["term_reason"][done].astype(np.int64))
            self.laps.append(infos["laps"][done].astype(np.int64))
            self.truncs.append(trunc[done])
            self.speeds.append(self.speed_sum[done] / np.maximum(self.speed_n[done], 1))
            self.speed_sum[done] = 0.0
            self.speed_n[done] = 0

    def pop(self) -> dict[str, float]:
        cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt)  # noqa: E731
        laps_t = cat(self.lap_times, np.float64)
        laps_t = laps_t[np.isfinite(laps_t)]
        reasons = cat(self.reasons, np.int64)
        laps = cat(self.laps, np.int64)
        truncs = cat(self.truncs, bool)
        speeds = cat(self.speeds, np.float64)
        lat = np.asarray(self.latencies) * 1e3
        n = len(reasons)
        m = {
            "race_episodes": float(n),
            "race_lap_time": float(laps_t.mean()) if len(laps_t) else math.nan,
            "race_lap_time_best": float(laps_t.min()) if len(laps_t) else math.nan,
            "race_completion_rate": float(truncs.mean()) if n else math.nan,
            "race_laps3_rate": float((laps >= 3).mean()) if n else math.nan,
            "race_laps": float(laps.mean()) if n else math.nan,
            "race_mean_speed": float(speeds.mean()) if n else math.nan,
            "step_latency_p50_ms": float(np.percentile(lat, 50)) if len(lat) else math.nan,
            "step_latency_p99_ms": float(np.percentile(lat, 99)) if len(lat) else math.nan,
        }
        for code, name in REASON_NAMES.items():
            m[f"race_term_{name.lower()}"] = float((reasons == int(code)).mean()) if n else math.nan
        self.clear()
        return m


class InstrumentedVecEnv:
    """
    Thin proxy around a SAME_STEP VectorEnv: forwards step_async / step_wait unchanged (collect_rollout keeps its
    fast path) and feeds every reply into RaceStats together with the vector-step latency.
    """

    def __init__(self, envs, stats: RaceStats):
        self.envs, self.stats = envs, stats
        self.num_envs = envs.num_envs
        self.single_observation_space = envs.single_observation_space
        self.single_action_space = envs.single_action_space
        self._t0 = 0.0

    @property
    def hello(self) -> dict:
        return getattr(self.envs, "hello", None) or self.envs.envs[0].hello

    def reset(self, **kw):
        return self.envs.reset(**kw)

    def step_async(self, actions) -> None:
        self._t0 = time.perf_counter()
        self.envs.step_async(actions)

    def step_wait(self):
        obs, rew, term, trunc, infos = self.envs.step_wait()
        self.stats.on_step(np.asarray(term, bool), np.asarray(trunc, bool), infos, time.perf_counter() - self._t0)
        return obs, rew, term, trunc, infos


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


class RunLogger:
    """TensorBoard (tags above) + metrics.jsonl / metrics.csv (one row per update) + eval.jsonl."""

    def __init__(self, run_dir: str | Path, purge_step: int | None = None, tensorboard: bool = True):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.dir / "tb"), purge_step=purge_step)
        self._csv_cols: list[str] | None = None
        csv_path = self.dir / "metrics.csv"
        if purge_step is not None and csv_path.exists():
            _truncate_rows(csv_path, self.dir / "metrics.jsonl", self.dir / "eval.jsonl", purge_step)
            with open(csv_path, encoding="utf-8", newline="") as f:
                self._csv_cols = next(csv.reader(f), None)

    def log_update(self, m: dict) -> None:
        step = int(m["global_step"])
        if self.writer is not None:
            for k, tag in TAGS.items():
                if _finite(m.get(k)):
                    self.writer.add_scalar(tag, m[k], step)
        with open(self.dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({k: (v if _finite(v) or not isinstance(v, float) else None) for k, v in m.items()})
                    + "\n")
        path = self.dir / "metrics.csv"
        new = self._csv_cols is None
        if new:
            self._csv_cols = sorted(k for k, v in m.items() if isinstance(v, (int, float)))
        with open(path, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(self._csv_cols)
            w.writerow([_fmt(m.get(c)) for c in self._csv_cols])

    def log_eval(self, step: int, per_seed: dict, extra: dict | None = None) -> None:
        if self.writer is not None:
            for k, tag in EVAL_TAGS.items():
                if _finite(per_seed.get(k)):
                    self.writer.add_scalar(tag, per_seed[k], step)
        row = {"global_step": step, **(extra or {}), **{k: v for k, v in per_seed.items() if k != "flying_laps_s"}}
        with open(self.dir / "eval.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return repr(float(v)) if isinstance(v, float) else str(v)


def _truncate_rows(csv_path: Path, jsonl_path: Path, eval_path: Path, purge_step: int) -> None:
    """Resume: drop rows written after the checkpoint we resume from (they will be re-generated)."""
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if rows:
        col = rows[0].index("global_step")
        keep = [rows[0]] + [r for r in rows[1:] if r[col] and float(r[col]) <= purge_step]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(keep)
    for p in (jsonl_path, eval_path):
        if p.exists():
            lines = [ln for ln in p.read_text(encoding="utf-8").splitlines()
                     if ln.strip() and json.loads(ln)["global_step"] <= purge_step]
            p.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
