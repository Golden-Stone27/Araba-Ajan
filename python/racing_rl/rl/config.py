"""PPOConfig (frozen dataclass), the "default" / "parity" presets, YAML loading and linear schedules (M4)."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

import yaml

ACTIVATIONS = ("tanh", "silu")
ACTION_TRANSFORMS = ("clip", "mlagents_scale")


@dataclass(frozen=True)
class PPOConfig:
    """
    All PPO hyperparameters. Scheduled values go linearly from `x` to `x_end` over progress ∈ [0, 1]
    (progress = global_step / total_steps); `x_end == x` means constant.
    """

    preset: str = "default"
    # network
    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: str = "tanh"
    action_transform: str = "clip"
    log_std_init: float = -0.5
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    norm_obs: bool = True
    obs_clip: float = 10.0
    # optimisation
    lr: float = 3e-4
    lr_end: float = 0.0
    adam_eps: float = 1e-5
    clip_eps: float = 0.2
    clip_eps_end: float = 0.2
    ent_coef: float = 1e-3
    ent_coef_end: float = 1e-3
    vf_coef: float = 0.5
    clip_vloss: bool = True
    vf_clip: float = 0.2
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    # returns
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # batching
    epochs: int = 4
    minibatch_size: int = 1024
    rollout_length: int = 256
    rollout_total: int | None = None  # set (parity): T = rollout_total // N_total, overrides rollout_length
    # runtime
    torch_threads_collect: int = 2
    torch_threads_update: int = 4

    def __post_init__(self):
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {self.activation!r}")
        if self.action_transform not in ACTION_TRANSFORMS:
            raise ValueError(f"action_transform must be one of {ACTION_TRANSFORMS}, got {self.action_transform!r}")
        if not self.log_std_min < self.log_std_max:
            raise ValueError("log_std_min must be < log_std_max")
        for name in ("epochs", "minibatch_size", "rollout_length", "torch_threads_collect", "torch_threads_update"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.rollout_total is not None and self.rollout_total < 1:
            raise ValueError("rollout_total must be >= 1 or None")
        object.__setattr__(self, "hidden_sizes", tuple(int(h) for h in self.hidden_sizes))

    def rollout_steps(self, num_envs: int) -> int:
        """Rollout length T for N_total = num_envs agents (parity: 20480 // N, e.g. 640 for 32 agents)."""
        if self.rollout_total is None:
            return self.rollout_length
        t = self.rollout_total // num_envs
        if t < 1:
            raise ValueError(f"rollout_total {self.rollout_total} < num_envs {num_envs}")
        return t

    def replace(self, **changes) -> PPOConfig:
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        """Primitives only (tuples → lists), so it can live inside a weights_only checkpoint and a YAML file."""
        d = dataclasses.asdict(self)
        d["hidden_sizes"] = list(self.hidden_sizes)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PPOConfig:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown PPOConfig keys: {sorted(unknown)}")
        return cls(**d)


presets: dict[str, PPOConfig] = {
    "default": PPOConfig(),
    # ML-Agents race_ppo.yaml equivalent (M2 baseline).
    "parity": PPOConfig(
        preset="parity", activation="silu", action_transform="mlagents_scale", log_std_init=0.0,
        lr=3e-4, lr_end=1e-10, clip_eps=0.2, clip_eps_end=0.1, ent_coef=5e-3, ent_coef_end=1e-5,
        vf_coef=0.5, clip_vloss=True, vf_clip=0.2, gamma=0.99, gae_lambda=0.95, epochs=3, minibatch_size=1024,
        rollout_total=20480,
    ),
}


def get_preset(name: str, **overrides) -> PPOConfig:
    if name not in presets:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(presets)}")
    return presets[name].replace(**overrides) if overrides else presets[name]


def load_yaml(path: str | os.PathLike) -> PPOConfig:
    """
    YAML mapping (optionally nested under `ppo:`). `preset:` picks the base (default "default"); every other key
    overrides a PPOConfig field. Unknown keys raise.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "ppo" in data and isinstance(data["ppo"], dict):
        data = data["ppo"]
    data = dict(data)
    base = presets[data.pop("preset", "default")]
    merged = base.to_dict()
    unknown = set(data) - set(merged)
    if unknown:
        raise ValueError(f"{path}: unknown PPOConfig keys: {sorted(unknown)}")
    merged.update(data)
    return PPOConfig.from_dict(merged)


class LinearSchedule:
    """value(progress) = start + (end − start) · clamp(progress, 0, 1)."""

    def __init__(self, start: float, end: float):
        self.start, self.end = float(start), float(end)

    def __call__(self, progress: float) -> float:
        p = min(max(float(progress), 0.0), 1.0)
        return self.start + (self.end - self.start) * p

    def __repr__(self) -> str:
        return f"LinearSchedule({self.start:g} -> {self.end:g})"
