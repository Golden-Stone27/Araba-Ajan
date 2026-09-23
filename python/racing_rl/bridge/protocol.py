"""M3 wire format (docs/milestones/M3_bridge_gym.md). Must stay byte-identical to Racing.Bridge.BridgeProtocol (C#)."""

from __future__ import annotations

import enum
import hashlib
import struct
from dataclasses import dataclass

import numpy as np

HEADER = struct.Struct("<IHHII")  # magic, version, msg_type, seq, payload_len
RESET = struct.Struct("<qII")  # seed, start_mode, max_laps
MAGIC = 0x47414352  # bytes "RCAG"
VERSION = 1
PROTOCOL = 1

MSG_HELLO = 0x0001
MSG_CONFIG = 0x0002
MSG_READY = 0x0003
MSG_RESET = 0x0010
MSG_STEP = 0x0011
MSG_STATE = 0x0020
MSG_CLOSE = 0x0030
MSG_ERROR = 0x00FF

MSG_NAMES = {
    MSG_HELLO: "HELLO", MSG_CONFIG: "CONFIG", MSG_READY: "READY", MSG_RESET: "RESET",
    MSG_STEP: "STEP", MSG_STATE: "STATE", MSG_CLOSE: "CLOSE", MSG_ERROR: "ERROR",
}

OBS_DIM = 26
ACT_DIM = 2
INFO_STRUCT = "RACE_INFO_V1"

# C0.6 canonical layout string; obs_layout_hash = sha256(utf8)[:16].
OBS_LAYOUT_V1 = (
    "RACE_OBS_V1|n=26|rays=15,fov=180,max=50,h=0.5|vfwd/50|vlat/50|yaw/3|psi1(sin,cos)@0|psi2(sin,cos)@30"
    "|elat/6|prev(steer,thr)|grounded/4"
)

# Frozen values (contracts.md C0.14).
FROZEN_OBS_LAYOUT_HASH = "b40ca79bdba1c2c2"
FROZEN_ENV_CONFIG_HASH = "90240ee2b1a58b5b"


def layout_hash(layout: str = OBS_LAYOUT_V1) -> str:
    return hashlib.sha256(layout.encode("utf-8")).hexdigest()[:16]


def _obs_bounds() -> tuple[np.ndarray, np.ndarray]:
    low = np.full(OBS_DIM, -1.0, np.float32)
    high = np.full(OBS_DIM, 1.0, np.float32)
    low[0:15] = 0.0  # rays
    low[22], high[22] = -1.5, 1.5  # e_lat
    low[25] = 0.0  # grounded
    return low, high


OBS_LOW, OBS_HIGH = _obs_bounds()

# RACE_INFO_V1, 40 B, '<BBHHHifffffff'
RACE_INFO_DTYPE = np.dtype(
    [
        ("term_reason", "u1"),
        ("lap_completed", "u1"),
        ("laps", "<u2"),
        ("next_cp", "<u2"),
        ("reserved", "<u2"),
        ("ep_decisions", "<i4"),
        ("last_lap_s", "<f4"),
        ("best_lap_s", "<f4"),
        ("speed_mps", "<f4"),
        ("progress", "<f4"),
        ("ep_return", "<f4"),
        ("pos_x", "<f4"),
        ("pos_z", "<f4"),
    ]
)
INFO_FIELDS = [n for n in RACE_INFO_DTYPE.names if n != "reserved"]
assert RACE_INFO_DTYPE.itemsize == 40 == struct.calcsize("<BBHHHifffffff")


class TermReason(enum.IntEnum):
    NONE = 0
    WALL = 1
    WRONG_WAY = 2
    STUCK = 3
    FLIP = 4
    OUT_OF_BOUNDS = 5
    TIME_LIMIT = 6
    PHYSICS_ERROR = 7
    FINISHED = 8


TERM_REASON_KEYS = {
    TermReason.NONE: "none", TermReason.WALL: "wall", TermReason.WRONG_WAY: "wrong_way", TermReason.STUCK: "stuck",
    TermReason.FLIP: "flip", TermReason.OUT_OF_BOUNDS: "out_of_bounds", TermReason.TIME_LIMIT: "time_limit",
    TermReason.PHYSICS_ERROR: "physics_error", TermReason.FINISHED: "finished",
}


class StartMode(enum.IntEnum):
    TRAIN_RANDOM = 0
    EVAL_GRID = 1


def flag_pad(n: int) -> int:
    return (4 - (2 * n) % 4) % 4


@dataclass(frozen=True)
class StateLayout:
    """Byte offsets of the STATE payload for N agents."""

    n: int

    @property
    def obs(self) -> int:
        return 0

    @property
    def reward(self) -> int:
        return self.n * OBS_DIM * 4

    @property
    def terminated(self) -> int:
        return self.reward + 4 * self.n

    @property
    def truncated(self) -> int:
        return self.terminated + self.n

    @property
    def final_obs(self) -> int:
        return self.truncated + self.n + flag_pad(self.n)

    @property
    def info(self) -> int:
        return self.final_obs + self.n * OBS_DIM * 4

    @property
    def size(self) -> int:
        return self.info + self.n * RACE_INFO_DTYPE.itemsize


@dataclass
class StateView:
    """Zero-copy numpy views into a received STATE payload (valid until the next recv)."""

    obs: np.ndarray  # f32[N, O]
    reward: np.ndarray  # f32[N]
    terminated: np.ndarray  # u8[N]
    truncated: np.ndarray  # u8[N]
    final_obs: np.ndarray  # f32[N, O]
    info: np.ndarray  # RACE_INFO_DTYPE[N]


def decode_state(payload, n: int) -> StateView:
    lay = StateLayout(n)
    if len(payload) != lay.size:
        raise ValueError(f"STATE payload {len(payload)} B != {lay.size} B for N={n}")
    return StateView(
        obs=np.frombuffer(payload, np.float32, n * OBS_DIM, lay.obs).reshape(n, OBS_DIM),
        reward=np.frombuffer(payload, np.float32, n, lay.reward),
        terminated=np.frombuffer(payload, np.uint8, n, lay.terminated),
        truncated=np.frombuffer(payload, np.uint8, n, lay.truncated),
        final_obs=np.frombuffer(payload, np.float32, n * OBS_DIM, lay.final_obs).reshape(n, OBS_DIM),
        info=np.frombuffer(payload, RACE_INFO_DTYPE, n, lay.info),
    )


def encode_state(obs, reward, terminated, truncated, final_obs, info, out: bytearray | None = None) -> bytearray:
    """Inverse of decode_state (used by mock_unity and tests)."""
    n = len(reward)
    lay = StateLayout(n)
    buf = out if out is not None and len(out) == lay.size else bytearray(lay.size)
    mv = memoryview(buf)
    mv[lay.obs:lay.reward] = np.ascontiguousarray(obs, np.float32).tobytes()
    mv[lay.reward:lay.terminated] = np.ascontiguousarray(reward, np.float32).tobytes()
    mv[lay.terminated:lay.truncated] = np.ascontiguousarray(terminated, np.uint8).tobytes()
    mv[lay.truncated:lay.truncated + n] = np.ascontiguousarray(truncated, np.uint8).tobytes()
    mv[lay.truncated + n:lay.final_obs] = bytes(flag_pad(n))
    mv[lay.final_obs:lay.info] = np.ascontiguousarray(final_obs, np.float32).tobytes()
    mv[lay.info:lay.size] = np.ascontiguousarray(info, RACE_INFO_DTYPE).tobytes()
    return buf
