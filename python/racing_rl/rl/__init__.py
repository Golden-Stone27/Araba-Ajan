"""M4: from-scratch PyTorch PPO (docs/milestones/M4_pytorch_ppo.md). M5 builds the training CLI on top of it."""

from .buffer import Minibatch, RolloutBuffer
from .config import LinearSchedule, PPOConfig, get_preset, load_yaml, presets
from .gae import compute_gae, gae_reference
from .loop import eval_rank, train
from .networks import ActorCritic
from .normalization import RunningMeanStd
from .policy import (CheckpointMismatchError, TorchPolicy, actor_critic_from_checkpoint, load_checkpoint, restore_rng,
                     save_checkpoint)
from .ppo import PPOTrainer
from .rollout import EpisodeStats, collect_rollout
from .synthetic_env import TargetReachVecEnv

__all__ = [
    "ActorCritic", "CheckpointMismatchError", "EpisodeStats", "LinearSchedule", "Minibatch", "PPOConfig", "PPOTrainer",
    "RolloutBuffer", "RunningMeanStd", "TargetReachVecEnv", "TorchPolicy", "actor_critic_from_checkpoint",
    "collect_rollout", "compute_gae", "eval_rank", "gae_reference", "get_preset", "load_checkpoint", "load_yaml",
    "presets", "restore_rng", "save_checkpoint", "train",
]
