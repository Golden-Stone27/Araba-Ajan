"""M3 bridge: Python TCP server <-> Unity client (docs/milestones/M3_bridge_gym.md)."""

from .errors import (BridgeError, DesyncError, ProtocolMismatchError, RemoteError, UnityCrashedError, UnityLaunchError,
                     UnityTimeoutError)
from .gym_env import RacingEnv
from .multi_env import MultiUnityVecEnv
from .policies import OnnxPolicy, Policy, RandomPolicy, ZeroPolicy
from .protocol import RACE_INFO_DTYPE, StartMode, TermReason, layout_hash
from .vec_env import UnityVecEnv

__all__ = [
    "BridgeError", "DesyncError", "MultiUnityVecEnv", "OnnxPolicy", "Policy", "ProtocolMismatchError", "RACE_INFO_DTYPE",
    "RacingEnv", "RandomPolicy", "RemoteError", "StartMode", "TermReason", "UnityCrashedError", "UnityLaunchError",
    "UnityTimeoutError", "UnityVecEnv", "ZeroPolicy", "layout_hash",
]
