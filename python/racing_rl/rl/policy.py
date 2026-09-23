"""
TorchPolicy (bridge Policy protocol: obs f32[N, 26] → actions f32[N, 2]) and checkpoint save / load.

Checkpoints hold tensors and primitives only, so `torch.load(path, weights_only=True)` reads them:
{model, optimizer, obs_rms{mean,var,count}, config, global_step, env_config_hash, obs_layout_hash, git_sha,
 rng{torch, numpy}, obs_dim, act_dim, extra}.
"""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from .config import PPOConfig
from .networks import ActorCritic

CHECKPOINT_FORMAT = "racing_rl.ppo/v1"


class CheckpointMismatchError(ValueError):
    """Checkpoint was trained on a different environment / observation layout (C0.11)."""


class TorchPolicy:
    """
    Deterministic by default (C0.10 eval: a = transform(μ)). Never updates the running observation statistics.
    `__call__(obs)` is the M3 bridge Policy protocol (racing_rl.bridge.evaluate.run_eval), `act(obs, deterministic)`
    the M4 one.
    """

    def __init__(self, ac: ActorCritic, deterministic: bool = True):
        self.ac = ac
        self.deterministic = deterministic

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if deterministic:
            return self.ac.act_deterministic(obs)
        return self.ac.act_full(obs, update_rms=False)[0]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.act(obs, self.deterministic)

    @classmethod
    def from_checkpoint(cls, path: str | os.PathLike, deterministic: bool = True,
                        expected_env_hash: str | None = None, expected_obs_hash: str | None = None) -> TorchPolicy:
        ckpt = load_checkpoint(path, expected_env_hash, expected_obs_hash)
        return cls(actor_critic_from_checkpoint(ckpt), deterministic)


@functools.lru_cache(maxsize=1)
def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True,
                             text=True, timeout=10)
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and sha else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def save_checkpoint(path: str | os.PathLike, ac: ActorCritic, optimizer: torch.optim.Optimizer | None, cfg: PPOConfig,
                    global_step: int, env_config_hash: str | None, obs_layout_hash: str | None, extra: dict | None = None,
                    np_rng: np.random.Generator | None = None) -> Path:
    """Atomic write (tmp + os.replace). `extra` must itself be weights_only-safe (tensors / primitives)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "format": CHECKPOINT_FORMAT,
        "model": ac.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "obs_rms": ac.obs_rms.state_dict(),
        "config": cfg.to_dict(),
        "obs_dim": ac.obs_dim,
        "act_dim": ac.act_dim,
        "global_step": int(global_step),
        "env_config_hash": env_config_hash,
        "obs_layout_hash": obs_layout_hash,
        "git_sha": git_sha(),
        "rng": {"torch": torch.get_rng_state(), "numpy": np_rng.bit_generator.state if np_rng is not None else None},
        "extra": extra or {},
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | os.PathLike, expected_env_hash: str | None = None,
                    expected_obs_hash: str | None = None) -> dict:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: not a {CHECKPOINT_FORMAT} checkpoint (format={ckpt.get('format')!r})")
    if expected_env_hash is not None and ckpt["env_config_hash"] != expected_env_hash:
        raise CheckpointMismatchError(
            f"{path}: env_config_hash {ckpt['env_config_hash']} != expected {expected_env_hash}")
    if expected_obs_hash is not None and ckpt["obs_layout_hash"] != expected_obs_hash:
        raise CheckpointMismatchError(
            f"{path}: obs_layout_hash {ckpt['obs_layout_hash']} != expected {expected_obs_hash}")
    return ckpt


def actor_critic_from_checkpoint(ckpt: dict) -> ActorCritic:
    """Rebuild the ActorCritic (architecture from the stored config) with weights and obs statistics."""
    cfg = PPOConfig.from_dict(ckpt["config"])
    ac = ActorCritic.from_config(cfg, int(ckpt["obs_dim"]), int(ckpt["act_dim"]))
    ac.load_state_dict(ckpt["model"])
    ac.obs_rms.load_state_dict(ckpt["obs_rms"])
    return ac


def restore_rng(ckpt: dict, np_rng: np.random.Generator | None = None) -> None:
    torch.set_rng_state(ckpt["rng"]["torch"])
    if np_rng is not None and ckpt["rng"]["numpy"] is not None:
        np_rng.bit_generator.state = ckpt["rng"]["numpy"]
