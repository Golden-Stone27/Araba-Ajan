"""M3 bridge: Python TCP server <-> Unity client."""

from .errors import (BridgeError, DesyncError, ProtocolMismatchError, RemoteError, UnityCrashedError, UnityLaunchError,
                     UnityTimeoutError)
from .gym_env import RacingEnv
from .multi_env import MultiUnityVecEnv
from .policies import OnnxPolicy, Policy, RandomPolicy, ZeroPolicy
from .protocol import RACE_INFO_DTYPE, StartMode, TermReason, layout_hash
from .tracks import TrackSpec, UnknownTrackError, composite_hash, load_catalog, resolve_track
from .vec_env import AUTO, UnityVecEnv

__all__ = [
    "AUTO", "BridgeError", "DesyncError", "MultiUnityVecEnv", "OnnxPolicy", "Policy", "ProtocolMismatchError",
    "RACE_INFO_DTYPE", "RacingEnv", "RandomPolicy", "RemoteError", "StartMode", "TermReason", "TrackSpec",
    "UnityCrashedError", "UnityLaunchError", "UnityTimeoutError", "UnityVecEnv", "UnknownTrackError", "ZeroPolicy",
    "composite_hash", "layout_hash", "load_catalog", "resolve_track",
]
