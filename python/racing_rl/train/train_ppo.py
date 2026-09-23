"""
M5 training CLI: custom PyTorch PPO (racing_rl.rl) on K Unity processes × N agents over the M3 bridge.

    python -m racing_rl.train.train_ppo --config configs/ppo_parity.yaml --seed 1 --run-dir ../runs/m5/parity_s1
    python -m racing_rl.train.train_ppo --run-dir ../runs/m5/parity_s1 --resume

Loop (M5 doc): progress = global_step / total_steps → collect T×N decisions (collect_rollout, torch threads 2) →
GAE → PPOTrainer.update (threads 4). Every ckpt_every updates ckpt_{step}.pt (≤ 10 kept); every eval_every updates
a C0.10 evaluation on a separate, persistent Unity process with the *validation* seeds (C0.19: 2000..2019) → best.pt.
Milestone snapshots (default 1M decisions) are kept for the sample-efficiency comparison.

Failure handling: a Unity crash / timeout / desync discards the running rollout, relaunches the dead process(es),
re-resets every process and continues (≤ 3 restarts per hour, otherwise the run stops with a checkpoint).
Ctrl+C: finish the current update, write a checkpoint, close every Unity process (kill_tree); a second Ctrl+C
aborts immediately. --resume continues from the newest ckpt_*.pt of the run directory (model, Adam, obs RMS,
trainer state, RNGs, milestones); TensorBoard / CSV rows after that step are purged.

Run directory: run.json (args, git sha, hashes), config.yaml (PPOConfig), checkpoints/, best.pt, milestone_*.pt,
metrics.{jsonl,csv}, eval.jsonl, tb/, unity_logs/.

Tracks (M6, contracts C0.20): --track T (repeatable) or run.tracks in the YAML; process k runs tracks[k % len] for
its whole life (MultiUnityVecEnv). Validation runs on run.eval_track (default: the first track). Checkpoints and
run.json carry env_config_hash = that track's hash (one track) or the composite "multi:<sha>" (several tracks,
racing_rl.bridge.tracks.composite_hash) and extra.tracks = {id: env_config_hash}. No --track is the M5 run (Track_A).

    python -m racing_rl.train.train_ppo --config configs/ppo_parity.yaml --seed 1 --run-dir ../runs/m6/mix_s1 \\
        --track Track_A --track Track_B --track Track_C --track Track_D
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import signal
import socket
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

from racing_rl.bridge.errors import BridgeError, UnityCrashedError
from racing_rl.bridge.evaluate import per_seed_report, run_eval
from racing_rl.bridge.multi_env import MultiUnityVecEnv
from racing_rl.bridge.protocol import FROZEN_ENV_CONFIG_HASH, FROZEN_OBS_LAYOUT_HASH
from racing_rl.bridge.tracks import resolve_track
from racing_rl.bridge.vec_env import UnityVecEnv
from racing_rl.rl import (ActorCritic, EpisodeStats, PPOConfig, PPOTrainer, RolloutBuffer, TorchPolicy,
                          actor_critic_from_checkpoint, collect_rollout, eval_rank, load_checkpoint, restore_rng)
from racing_rl.rl.config import presets
from racing_rl.rl.policy import git_sha, save_checkpoint

from .telemetry import InstrumentedVecEnv, RaceStats, RunLogger

REPO = Path(__file__).resolve().parents[3]
DEFAULT_EXE = REPO / "Builds" / "RaceEnv" / "RaceEnv.exe"
CKPT_RE = re.compile(r"^ckpt_(\d+)\.pt$")
MAX_RESTARTS_PER_HOUR = 3

RUN_DEFAULTS = {
    "total_steps": 10_000_000, "procs": 3, "agents": 16, "eval_every": 25, "ckpt_every": 10, "max_checkpoints": 10,
    "eval_seed_base": 2000, "eval_episodes": 20, "milestones": [1_000_000], "base_port": 6005, "eval_port": 6205,
    "tracks": None, "eval_track": None,
}


# ---------------------------------------------------------------------------------------------------------- config
def load_run_config(path: str | None) -> tuple[PPOConfig, dict]:
    """YAML with an optional `ppo:` section (PPOConfig, `preset:` picks the base) and `run:` section (RUN_DEFAULTS)."""
    if path is None:
        return presets["parity"], {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    ppo = dict(data.get("ppo", {k: v for k, v in data.items() if k != "run"}))
    base = presets[ppo.pop("preset", "default")].to_dict()
    unknown = set(ppo) - set(base)
    if unknown:
        raise ValueError(f"{path}: unknown PPOConfig keys {sorted(unknown)}")
    base.update(ppo)
    run = {k: _num(v) for k, v in dict(data.get("run", {})).items()}
    unknown = set(run) - set(RUN_DEFAULTS)
    if unknown:
        raise ValueError(f"{path}: unknown run keys {sorted(unknown)}")
    return PPOConfig.from_dict({k: _num(v) for k, v in base.items()}), run


def _num(v):
    """PyYAML (YAML 1.1) reads '1.0e7' / '3e-4' as strings; coerce numeric-looking strings (recursively in lists)."""
    if isinstance(v, list):
        return [_num(x) for x in v]
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        out[k.strip()] = _num(yaml.safe_load(v))
    return out


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    found = sorted((int(m.group(1)), p) for p in ckpt_dir.glob("ckpt_*.pt") if (m := CKPT_RE.match(p.name)))
    return found[-1][1] if found else None


def prune(ckpt_dir: Path, keep: int) -> None:
    found = sorted((int(m.group(1)), p) for p in ckpt_dir.iterdir() if (m := CKPT_RE.match(p.name)))
    for _, p in found[:max(len(found) - keep, 0)]:
        p.unlink()


def normalize_tracks(run: dict) -> None:
    """Canonical track ids (unknown names fail before any Unity process starts); eval_track defaults to tracks[0]."""
    tracks = run.get("tracks")
    if isinstance(tracks, str):
        tracks = [tracks]
    run["tracks"] = [resolve_track(t).id for t in tracks] if tracks else None
    ev = run.get("eval_track")
    run["eval_track"] = resolve_track(ev).id if ev else (run["tracks"][0] if run["tracks"] else None)


def reset_seed(seed: int, update: int, attempt: int = 0) -> int:
    """Fresh run: the training seed itself; resumes / restarts get a distinct (still reproducible) spawn stream."""
    return seed if update == 0 and attempt == 0 else seed + 1_000_003 * update + 7_919 * attempt


# ---------------------------------------------------------------------------------------------------------- env
def recover(envs: MultiUnityVecEnv, seed: int) -> np.ndarray:
    """After a bridge failure: drain in-flight replies, relaunch dead processes, re-reset everything."""
    for e in envs.envs:
        if e.connected and e._pending:
            try:
                e._recv_state()
            except BridgeError:
                pass
        if not e.connected:
            e.restart()
    return np.asarray(envs.reset(seed=seed)[0], np.float32)


def validate(eval_env: UnityVecEnv, ac: ActorCritic, train_seed: int, seed_base: int, episodes: int) -> dict:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        eps = run_eval(eval_env, TorchPolicy(ac, deterministic=True), episodes=episodes, seed_base=seed_base)
    finally:
        torch.set_num_threads(threads)
    return per_seed_report(eps, train_seed)


class GracefulStop:
    """First Ctrl+C: stop after the current update (checkpoint + clean shutdown). Second: KeyboardInterrupt."""

    def __init__(self):
        self.requested = False
        self._prev = signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum, frame):
        if self.requested:
            signal.signal(signal.SIGINT, self._prev)
            raise KeyboardInterrupt
        self.requested = True
        print("\n[train] Ctrl+C: stopping after this update (press again to abort)", flush=True)

    def restore(self) -> None:
        signal.signal(signal.SIGINT, self._prev)


# ---------------------------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML (ppo: / run: sections); ignored with --resume")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--exe", default=str(DEFAULT_EXE))
    for k in ("total_steps", "procs", "agents", "eval_every", "ckpt_every", "eval_seed_base", "eval_episodes",
              "base_port", "eval_port"):
        ap.add_argument("--" + k.replace("_", "-"), type=float if k == "total_steps" else int)
    ap.add_argument("--milestones", type=float, nargs="*")
    ap.add_argument("--track", dest="tracks", action="append", metavar="TRACK",
                    help="catalog id or proc:<seed>; repeat for mixed-track training (process k: tracks[k %% len])")
    ap.add_argument("--eval-track", help="validation track (default: the first --track)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="PPOConfig overrides")
    ap.add_argument("--max-updates", type=int, help="stop after this many updates in this session (tests)")
    ap.add_argument("--no-tensorboard", action="store_true")
    a = ap.parse_args(argv)

    exe = a.exe or None  # --exe "" : wait for a connecting Editor / mock_unity (tests)
    run_dir = Path(a.run_dir).resolve()
    ckpt_dir = run_dir / "checkpoints"
    ckpt = None
    if a.resume:
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        ckpt_path = latest_checkpoint(ckpt_dir)
        if ckpt_path is None:
            raise SystemExit(f"--resume: no ckpt_*.pt in {ckpt_dir}")
        ckpt = load_checkpoint(ckpt_path, meta.get("env_config_hash") or FROZEN_ENV_CONFIG_HASH, FROZEN_OBS_LAYOUT_HASH)
        cfg = PPOConfig.from_dict(ckpt["config"])
        run = dict(meta["run"])
        run.setdefault("tracks", None)  # M5 run directories predate the track keys
        run.setdefault("eval_track", None)
        seed = int(meta["seed"])
        for k in ("base_port", "eval_port"):  # the only knobs a resume may change
            if getattr(a, k) is not None:
                run[k] = getattr(a, k)
        print(f"[train] resuming {run_dir.name} from {ckpt_path.name}", flush=True)
    else:
        if latest_checkpoint(ckpt_dir) is not None:
            raise SystemExit(f"{run_dir} already has checkpoints; use --resume or a new --run-dir")
        cfg, run_yaml = load_run_config(a.config)
        if a.set:
            cfg = cfg.replace(**parse_overrides(a.set))
        run = {**RUN_DEFAULTS, **run_yaml}
        for k in RUN_DEFAULTS:
            v = getattr(a, k, None)
            if v is not None and (k != "milestones" or v):
                run[k] = v
        run["total_steps"] = int(run["total_steps"])
        run["milestones"] = [int(x) for x in run["milestones"]]
        normalize_tracks(run)
        seed = int(a.seed if a.seed is not None else 1)
        meta = {"seed": seed, "run": run, "config_file": a.config, "overrides": a.set, "git_sha": git_sha(),
                "env_config_hash": None, "obs_layout_hash": None, "exe": a.exe, "host": socket.gethostname(),
                "created": dt.datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
                "torch": torch.__version__}
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump({"ppo": cfg.to_dict(), "run": run}, sort_keys=False),
                                             encoding="utf-8")

    # ---- seeding (same order as racing_rl.rl.loop.train so both paths give the same first update)
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    envs = eval_env = logger = None
    stop = GracefulStop()
    status = "error"
    try:
        envs = MultiUnityVecEnv(exe, num_processes=run["procs"], num_agents=run["agents"], base_port=run["base_port"],
                                log_dir=run_dir / "unity_logs", tracks=run["tracks"])
        hello = envs.envs[0].hello
        env_hash = envs.env_config_hash  # the track's hash, or multi:<sha> over several tracks
        track_hashes = dict(envs.env_hashes)
        if not a.resume:
            meta.update(env_config_hash=env_hash, tracks=track_hashes, obs_layout_hash=hello["obs_layout_hash"],
                        unity=hello["unity"], build_id=hello["build_id"])
            (run_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        elif env_hash != meta["env_config_hash"]:
            raise SystemExit(f"--resume: environment {env_hash} != run.json {meta['env_config_hash']}")
        if run["eval_every"] > 0:
            eval_env = UnityVecEnv(exe, num_agents=run["eval_episodes"], port=run["eval_port"],
                                   log_dir=run_dir / "unity_logs", track=run["eval_track"])

        N = envs.num_envs
        obs_dim = int(envs.single_observation_space.shape[0])
        act_dim = int(envs.single_action_space.shape[0])
        T = cfg.rollout_steps(N)
        num_updates = max(1, math.ceil(run["total_steps"] / (T * N)))

        ac = ActorCritic.from_config(cfg, obs_dim, act_dim) if ckpt is None else actor_critic_from_checkpoint(ckpt)
        trainer = PPOTrainer(ac, cfg, seed=seed)
        state = {"best_completion": None, "best_median": None, "best_step": None, "decisions_to_first_3lap": None,
                 "decisions_to_95pct": None, "wallclock_s": 0.0, "restarts": 0, "saved_milestones": []}
        update, global_step = 0, 0
        if ckpt is not None:
            trainer.optimizer.load_state_dict(ckpt["optimizer"])
            trainer.load_state_dict(ckpt["extra"]["trainer"])
            restore_rng(ckpt, np_rng)
            update, global_step = int(ckpt["extra"]["update"]), int(ckpt["global_step"])
            state.update(ckpt["extra"]["state"])

        stats = RaceStats(N)
        ienv = InstrumentedVecEnv(envs, stats)
        buf = RolloutBuffer(T, N, obs_dim, act_dim)
        ep_stats = EpisodeStats(N)
        logger = RunLogger(run_dir, purge_step=global_step if ckpt is not None else None,
                           tensorboard=not a.no_tensorboard)
        tracks_note = f", tracks {envs.track_ids} (eval {run['eval_track']})" if run["tracks"] else ""
        print(f"[train] {run_dir.name}: seed {seed}, preset {cfg.preset}, K={run['procs']} × N={run['agents']} "
              f"(N_total {N}), T={T}, {T * N} decisions/update, {num_updates} updates, "
              f"env {env_hash}{tracks_note}", flush=True)

        def checkpoint(path: Path, extra: dict | None = None) -> None:
            st = dict(state, wallclock_s=state["wallclock_s"] + (time.perf_counter() - t_session))
            save_checkpoint(path, ac, trainer.optimizer, cfg, global_step, env_hash, hello["obs_layout_hash"],
                            {"trainer": trainer.state_dict(), "update": update, "state": st, "tracks": track_hashes,
                             **(extra or {})}, np_rng)

        t_session = time.perf_counter()
        obs = np.asarray(envs.reset(seed=reset_seed(seed, update))[0], np.float32)
        restarts: list[float] = []
        session_updates = 0
        while update < num_updates:
            if stop.requested or (a.max_updates is not None and session_updates >= a.max_updates):
                break
            progress = global_step / run["total_steps"]
            t0 = time.perf_counter()
            try:
                obs, last_values = collect_rollout(ienv, ac, buf, obs, cfg, ep_stats)
            except BridgeError as e:
                now = time.monotonic()
                restarts = [t for t in restarts if now - t < 3600.0] + [now]
                state["restarts"] += 1
                print(f"[train] bridge failure during update {update + 1}: {type(e).__name__}: {e}", flush=True)
                if len(restarts) > MAX_RESTARTS_PER_HOUR:
                    raise UnityCrashedError(f"more than {MAX_RESTARTS_PER_HOUR} Unity restarts within an hour") from e
                obs = recover(envs, reset_seed(seed, update, attempt=state["restarts"]))
                stats.clear()
                stats.discard_running()
                ep_stats = EpisodeStats(N)
                print(f"[train] recovered (restart {len(restarts)} this hour); rollout discarded", flush=True)
                continue
            t1 = time.perf_counter()
            update += 1
            global_step += T * N
            buf.compute_gae(last_values, cfg.gamma, cfg.gae_lambda)
            m = trainer.update(buf, progress)
            t2 = time.perf_counter()
            ep_ret, ep_len = ep_stats.pop()
            m.update({
                "update": float(update), "global_step": float(global_step), "progress": progress,
                "episodes": float(len(ep_ret)),
                "ep_return_mean": float(np.mean(ep_ret)) if ep_ret else math.nan,
                "ep_length_mean": float(np.mean(ep_len)) if ep_len else math.nan,
                "rollout_reward_mean": float(buf.rewards.mean()),
                "time_collect_s": t1 - t0, "time_update_s": t2 - t1, "sps": T * N / (t2 - t0),
                "collect_sps": T * N / (t1 - t0), "entropy_per_dim": m["entropy"] / act_dim,
                "wallclock_s": state["wallclock_s"] + (t2 - t_session),
            })
            m.update(stats.pop())
            if state["decisions_to_first_3lap"] is None and m["race_laps3_rate"] > 0:
                state["decisions_to_first_3lap"] = global_step
            if state["decisions_to_95pct"] is None and m["race_completion_rate"] >= 0.95:
                state["decisions_to_95pct"] = global_step

            last = update == num_updates
            if update % run["ckpt_every"] == 0 or last:
                checkpoint(ckpt_dir / f"ckpt_{global_step}.pt")
                prune(ckpt_dir, run["max_checkpoints"])
            for ms in run["milestones"]:
                if global_step >= ms and ms not in state["saved_milestones"]:
                    state["saved_milestones"] = state["saved_milestones"] + [ms]
                    checkpoint(run_dir / f"milestone_{ms}.pt")

            if eval_env is not None and (update % run["eval_every"] == 0 or last):
                te = time.perf_counter()
                per_seed = None
                for attempt in range(2):
                    try:
                        per_seed = validate(eval_env, ac, seed, run["eval_seed_base"], run["eval_episodes"])
                        break
                    except BridgeError as e:
                        print(f"[train] eval bridge failure ({type(e).__name__}: {e}); relaunching eval env",
                              flush=True)
                        eval_env._res.shutdown(graceful=False)
                        eval_env._pending = False
                        eval_env._launch()
                if per_seed is not None:
                    key = eval_rank(per_seed)
                    best_key = None if state["best_completion"] is None else (
                        state["best_completion"], -state["best_median"] if state["best_median"] is not None
                        else -math.inf)
                    is_best = best_key is None or key > best_key
                    m["eval_completion_rate"] = key[0]
                    m["eval_flying_lap_median_s"] = per_seed["flying_lap_median_s"] or math.nan
                    m["eval_best"] = float(is_best)
                    m["time_eval_s"] = time.perf_counter() - te
                    if is_best:
                        state["best_completion"] = key[0]
                        state["best_median"] = per_seed["flying_lap_median_s"]
                        state["best_step"] = global_step
                        checkpoint(run_dir / "best.pt", {"eval": per_seed})
                    eval_meta = {"update": update, "best": is_best, "seed_base": run["eval_seed_base"]}
                    if run["eval_track"]:
                        eval_meta["track"] = run["eval_track"]
                    logger.log_eval(global_step, per_seed, eval_meta)
            logger.log_update(m)
            session_updates += 1
            _print_update(m, num_updates)
            if update % 10 == 0:
                logger.flush()

        if stop.requested or update < num_updates:
            checkpoint(ckpt_dir / f"ckpt_{global_step}.pt")
            prune(ckpt_dir, run["max_checkpoints"])
            status = "stopped"
        else:
            status = "finished"
        state["wallclock_s"] += time.perf_counter() - t_session
        summary = {"status": status, "update": update, "num_updates": num_updates, "global_step": global_step,
                   **{k: state[k] for k in ("best_completion", "best_median", "best_step", "decisions_to_first_3lap",
                                            "decisions_to_95pct", "wallclock_s", "restarts")}}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[train] {status}: {json.dumps(summary)}", flush=True)
        return 0
    except KeyboardInterrupt:
        status = "aborted"
        print("[train] aborted (second Ctrl+C); last periodic checkpoint is the resume point", flush=True)
        return 130
    finally:
        stop.restore()
        for e in (envs, eval_env):
            if e is not None:
                try:
                    e.close()
                except Exception as ex:  # noqa: BLE001 - shutdown must reach every process
                    warnings.warn(f"close failed: {ex}")
        if logger is not None:
            logger.close()


def _print_update(m: dict, num_updates: int) -> None:
    f = lambda k, fmt: format(m[k], fmt) if k in m and m[k] is not None and math.isfinite(m[k]) else "-"  # noqa: E731
    line = (f"[{int(m['update']):4d}/{num_updates}] step {int(m['global_step']):>9,d} "
            f"ret {f('ep_return_mean', '8.2f')} len {f('ep_length_mean', '6.0f')} "
            f"cmpl {f('race_completion_rate', '.2f')} 3lap {f('race_laps3_rate', '.2f')} "
            f"lap {f('race_lap_time', '.2f')} wall {f('race_term_wall', '.2f')} "
            f"kl {f('approx_kl', '.4f')} ev {f('explained_var', '.2f')} H {f('entropy', '.3f')} "
            f"sps {f('sps', ',.0f')} (col {f('collect_sps', ',.0f')}, upd {f('time_update_s', '.2f')}s, "
            f"p99 {f('step_latency_p99_ms', '.1f')}ms)")
    if "eval_completion_rate" in m:
        line += (f"\n       eval@2000: completion {m['eval_completion_rate']:.2f} "
                 f"flying {f('eval_flying_lap_median_s', '.2f')} s{' *best*' if m.get('eval_best') else ''}")
    print(line, flush=True)


if __name__ == "__main__":
    sys.exit(main())
