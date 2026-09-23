"""
Library-level PPO training loop (M5 wraps it with the CLI, TensorBoard, restarts and resume).

    progress = global_step / total_steps ; collect T×N decisions ; GAE ; PPO update
    every ckpt_every updates: ckpt_{global_step}.pt (at most max_checkpoints kept)
    every eval_every updates: eval_fn(TorchPolicy) → C0.10 per_seed dict → best.pt
global_step counts environment decisions (T·N per update), like ML-Agents' "Step".
"""

from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .buffer import RolloutBuffer
from .config import PPOConfig
from .networks import ActorCritic
from .policy import TorchPolicy, save_checkpoint
from .ppo import PPOTrainer
from .rollout import EpisodeStats, collect_rollout

_CKPT_RE = re.compile(r"^ckpt_(\d+)\.pt$")


def eval_rank(per_seed: dict) -> tuple[float, float]:
    """Sort key, larger is better: completion_rate desc, then flying_lap_median_s asc (None = worst)."""
    med = per_seed.get("flying_lap_median_s")
    return float(per_seed.get("completion_rate") or 0.0), -float(med) if med is not None else -math.inf


def _prune(ckpt_dir: Path, keep: int) -> None:
    found = sorted((int(m.group(1)), p) for p in ckpt_dir.iterdir() if (m := _CKPT_RE.match(p.name)))
    for _, p in found[:max(len(found) - keep, 0)]:
        p.unlink()


def train(envs, cfg: PPOConfig, total_steps: int, ckpt_dir: str | os.PathLike | None, eval_fn: Callable | None = None,
          eval_every: int = 25, ckpt_every: int = 10, seed: int = 0, log_fn: Callable[[dict], None] | None = None, *,
          ac: ActorCritic | None = None, env_config_hash: str | None = None, obs_layout_hash: str | None = None,
          max_checkpoints: int = 10) -> list[dict]:
    """
    Train `ac` (built from cfg when None; pass one to keep a handle) for ceil(total_steps / (T·N)) updates.
    Returns the per-update metric history. Env hashes default to envs.hello (UnityVecEnv) when available.
    """
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    N = int(envs.num_envs)
    obs_dim = int(envs.single_observation_space.shape[0])
    act_dim = int(envs.single_action_space.shape[0])
    T = cfg.rollout_steps(N)
    hello = getattr(envs, "hello", None) or getattr(getattr(envs, "envs", [None])[0], "hello", None) or {}
    env_config_hash = env_config_hash or hello.get("env_config_hash")
    obs_layout_hash = obs_layout_hash or hello.get("obs_layout_hash")

    if ac is None:
        ac = ActorCritic.from_config(cfg, obs_dim, act_dim)
    trainer = PPOTrainer(ac, cfg, seed=seed)
    buf = RolloutBuffer(T, N, obs_dim, act_dim)
    stats = EpisodeStats(N)
    ckpt_path = Path(ckpt_dir) if ckpt_dir is not None else None
    if ckpt_path is not None:
        ckpt_path.mkdir(parents=True, exist_ok=True)

    num_updates = max(1, math.ceil(total_steps / (T * N)))
    global_step = 0
    best_key: tuple[float, float] | None = None
    history: list[dict] = []
    obs, _ = envs.reset(seed=seed)
    obs = np.asarray(obs, np.float32)
    t_start = time.perf_counter()

    def checkpoint(name: str, extra: dict | None = None) -> None:
        save_checkpoint(ckpt_path / name, ac, trainer.optimizer, cfg, global_step, env_config_hash, obs_layout_hash,
                        {"trainer": trainer.state_dict(), "update": update, **(extra or {})}, np_rng)

    for update in range(1, num_updates + 1):
        progress = global_step / total_steps
        t0 = time.perf_counter()
        obs, last_values = collect_rollout(envs, ac, buf, obs, cfg, stats)
        t1 = time.perf_counter()
        global_step += T * N
        buf.compute_gae(last_values, cfg.gamma, cfg.gae_lambda)
        m = trainer.update(buf, progress)
        t2 = time.perf_counter()
        ep_ret, ep_len = stats.pop()
        m.update({
            "update": float(update), "global_step": float(global_step), "progress": progress,
            "episodes": float(len(ep_ret)),
            "ep_return_mean": float(np.mean(ep_ret)) if ep_ret else math.nan,
            "ep_length_mean": float(np.mean(ep_len)) if ep_len else math.nan,
            "rollout_reward_mean": float(buf.rewards.mean()),
            "time_collect_s": t1 - t0, "time_update_s": t2 - t1, "sps": T * N / (t2 - t0),
            "time_total_s": t2 - t_start,
        })

        last = update == num_updates
        if ckpt_path is not None and (update % ckpt_every == 0 or last):
            checkpoint(f"ckpt_{global_step}.pt")
            _prune(ckpt_path, max_checkpoints)
        if eval_fn is not None and update % eval_every == 0:
            per_seed = eval_fn(TorchPolicy(ac, deterministic=True))
            key = eval_rank(per_seed)
            m["eval_completion_rate"] = float(per_seed.get("completion_rate") or 0.0)
            med = per_seed.get("flying_lap_median_s")
            m["eval_flying_lap_median_s"] = float(med) if med is not None else math.nan
            m["eval_best"] = float(best_key is None or key > best_key)
            if best_key is None or key > best_key:
                best_key = key
                if ckpt_path is not None:
                    checkpoint("best.pt", {"eval": per_seed})
        history.append(m)
        if log_fn is not None:
            log_fn(m)
    return history
