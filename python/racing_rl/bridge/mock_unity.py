"""
Protocol-speaking stand-in for Unity (tests without a build): circular track + point-mass kinematics.
Implements the same STEP semantics as BridgeDriver.cs (K sub-steps, zero action after done, final_obs/final_info,
in-block reset, reward carry-over, lap flag OR-ed over sub-steps) and can inject faults.

    python -m racing_rl.bridge.mock_unity --port 6005 --num-agents 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import threading
import time

import numpy as np

from .protocol import (ACT_DIM, ERR_HASH_MISMATCH, ERR_TRACK_MISMATCH, FROZEN_ENV_CONFIG_HASH, HEADER, INFO_STRUCT, MAGIC,
                       MSG_CLOSE, MSG_CONFIG, MSG_ERROR, MSG_HELLO, MSG_READY, MSG_RESET, MSG_STATE, MSG_STEP, OBS_DIM,
                       PROTOCOL, RACE_INFO_DTYPE, RESET, VERSION, StartMode, StateLayout, TermReason, encode_state,
                       layout_hash)

RADIUS = 60.0
HALF_WIDTH = 6.0
LENGTH = 2.0 * math.pi * RADIUS
CHECKPOINTS = 16


class MockUnity:
    def __init__(self, port: int, num_agents: int = 4, host: str = "127.0.0.1", decision_period: int = 5,
                 fixed_dt: float = 0.02, max_episode_decisions: int = 3000, env_config_hash: str = FROZEN_ENV_CONFIG_HASH,
                 obs_layout_hash: str | None = None, desync_at_step: int | None = None, hang_at_step: int | None = None,
                 hang_s: float = 5.0, crash_at_step: int | None = None, track_id: str = "mock_circle",
                 track_index: int = -1):
        self.host, self.port, self.n, self.k, self.dt = host, port, num_agents, decision_period, fixed_dt
        self.max_dec = max_episode_decisions
        self.env_hash = env_config_hash
        self.obs_hash = obs_layout_hash or layout_hash()
        # HELLO track fields (M6); the kinematics stay the circle whatever the id says.
        self.track_id, self.track_index = track_id, track_index
        self.track_hash = hashlib.sha256(f"{track_id}|{RADIUS}|{HALF_WIDTH}".encode()).hexdigest()[:16]
        self.desync_at, self.hang_at, self.hang_s, self.crash_at = desync_at_step, hang_at_step, hang_s, crash_at_step
        self.steps = 0
        self.requests = 0
        self.error: BaseException | None = None
        n = num_agents
        self.seed, self.mode, self.max_laps = 0, StartMode.TRAIN_RANDOM, 0
        self.rng = [np.random.default_rng(i) for i in range(n)]
        # per-agent state
        self.s = np.zeros(n)
        self.d = np.zeros(n)
        self.psi = np.zeros(n)
        self.v = np.zeros(n)
        self.prev = np.zeros((n, ACT_DIM))
        self.steps_ep = np.zeros(n, np.int64)
        self.stuck = np.zeros(n, np.int64)
        self.laps = np.zeros(n, np.int64)
        self.lap_steps = np.zeros(n, np.int64)
        self.last_lap = np.full(n, np.nan)
        self.best_lap = np.full(n, np.nan)
        self.ep_return = np.zeros(n)
        self.reason = np.zeros(n, np.int64)
        self._state_buf = bytearray(StateLayout(n).size)

    # ------------------------------------------------------------------ dynamics
    def _spawn(self, i: int) -> None:
        r = self.rng[i]
        if self.mode == StartMode.EVAL_GRID:
            self.s[i], self.d[i], self.psi[i], self.v[i] = 1.0, r.uniform(-0.5, 0.5), math.radians(r.uniform(-2, 2)), 0.0
        else:
            self.s[i], self.d[i] = r.uniform(0, LENGTH), r.uniform(-1.8, 1.8)
            self.psi[i], self.v[i] = math.radians(r.uniform(-10, 10)), r.uniform(0, 10)
        self.prev[i] = 0.0
        self.steps_ep[i] = self.stuck[i] = self.laps[i] = self.lap_steps[i] = 0
        self.last_lap[i] = self.best_lap[i] = np.nan
        self.ep_return[i] = 0.0
        self.reason[i] = 0

    def _physics(self, i: int, a: np.ndarray) -> tuple[float, int, bool]:
        """One 0.02 s sub-step for agent i. Returns (reward, reason, lap_completed)."""
        steer, thr = float(np.clip(a[0], -1, 1)), float(np.clip(a[1], -1, 1))
        dt, v = self.dt, self.v[i]
        v += (thr * 8.0 if thr >= 0 else thr * 15.0) * dt - 0.02 * v * v * dt / 10.0
        v = min(max(v, 0.0), 45.0)
        self.psi[i] += (steer * 0.8 * min(v / 10.0, 1.0) - v / RADIUS) * dt
        self.d[i] += v * math.sin(self.psi[i]) * dt
        ds = v * math.cos(self.psi[i]) * dt
        self.v[i] = v
        self.steps_ep[i] += 1
        self.lap_steps[i] += 1
        lap = False
        s_new = self.s[i] + ds
        if s_new >= LENGTH:
            s_new -= LENGTH
            lap = True
            t = self.lap_steps[i] * dt
            self.last_lap[i] = t
            self.best_lap[i] = t if np.isnan(self.best_lap[i]) else min(t, self.best_lap[i])
            self.laps[i] += 1
            self.lap_steps[i] = 0
        self.s[i] = s_new % LENGTH
        self.stuck[i] = self.stuck[i] + 1 if ds < 0.01 else 0
        d_steer = abs(steer - self.prev[i][0])
        self.prev[i] = (steer, thr)

        reason = TermReason.NONE
        if abs(self.d[i]) > HALF_WIDTH:
            reason = TermReason.WALL
        elif self.stuck[i] * dt >= 8.0:
            reason = TermReason.STUCK
        elif self.max_laps > 0 and self.laps[i] >= self.max_laps:
            reason = TermReason.FINISHED
        elif self.steps_ep[i] >= self.max_dec * self.k:
            reason = TermReason.TIME_LIMIT
        r = 0.1 * v * math.cos(self.psi[i]) / 50.0 / self.k - 0.02 * d_steer
        if lap:
            r += 2.0
        if reason in (TermReason.WALL, TermReason.STUCK):
            r += -1.0 if reason == TermReason.WALL else -0.5
        self.ep_return[i] += np.float32(r)
        if reason:
            self.reason[i] = reason
        return float(np.float32(r)), int(reason), lap

    def _obs(self, i: int, dst: np.ndarray) -> None:
        o = dst[i]
        for j in range(15):
            ang = math.radians(-90 + j * 180 / 14) + self.psi[i]
            side = math.sin(ang)
            dist = (HALF_WIDTH - self.d[i]) / side if side > 1e-3 else ((-HALF_WIDTH - self.d[i]) / side if side < -1e-3 else 99.0)
            o[j] = min(max(dist / 50.0, 0.0), 1.0)
        v = self.v[i]
        o[15] = np.clip(v * math.cos(0) / 50.0, -1, 1)
        o[16] = 0.0
        o[17] = np.clip(-v / RADIUS / 3.0, -1, 1)
        o[18], o[19] = math.sin(-self.psi[i]), math.cos(-self.psi[i])
        o[20], o[21] = math.sin(-self.psi[i] + 0.5), math.cos(-self.psi[i] + 0.5)
        o[22] = np.clip(self.d[i] / 6.0, -1.5, 1.5)
        o[23], o[24] = self.prev[i]
        o[25] = 1.0

    def _info(self, i: int, info: np.ndarray, lap_flag: bool) -> None:
        rec = info[i]  # np.void view: field assignment writes through
        rec["term_reason"] = self.reason[i]
        rec["lap_completed"] = lap_flag
        rec["laps"] = self.laps[i]
        rec["next_cp"] = int(self.s[i] / LENGTH * CHECKPOINTS) % CHECKPOINTS
        rec["reserved"] = 0
        rec["ep_decisions"] = self.steps_ep[i] // self.k
        rec["last_lap_s"] = self.last_lap[i]
        rec["best_lap_s"] = self.best_lap[i]
        rec["speed_mps"] = self.v[i]
        rec["progress"] = self.s[i] / LENGTH
        rec["ep_return"] = self.ep_return[i]
        rec["pos_x"] = RADIUS * math.cos(self.s[i] / RADIUS)
        rec["pos_z"] = RADIUS * math.sin(self.s[i] / RADIUS)

    # ------------------------------------------------------------------ protocol
    def reset_all(self, seed: int, mode: int, max_laps: int) -> None:
        self.seed, self.mode, self.max_laps = seed, StartMode(mode), max_laps
        self.rng = [np.random.default_rng(seed + i) for i in range(self.n)]
        for i in range(self.n):
            self._spawn(i)
        self.carry = np.zeros(self.n)

    def state_payload(self, actions: np.ndarray | None) -> bytearray:
        n = self.n
        obs = np.zeros((n, OBS_DIM), np.float32)
        final_obs = np.zeros((n, OBS_DIM), np.float32)
        info = np.zeros(n, RACE_INFO_DTYPE)
        reward = np.zeros(n)
        term = np.zeros(n, np.uint8)
        trunc = np.zeros(n, np.uint8)
        lap = np.zeros(n, bool)
        done = np.zeros(n, bool)
        if actions is not None:
            reward[:] = self.carry
            self.carry[:] = 0
            for _ in range(self.k):
                for i in range(n):
                    r, reason, lp = self._physics(i, np.zeros(2) if done[i] else actions[i])
                    if done[i]:
                        self.carry[i] += r
                        if reason:
                            self.carry[i] = 0
                            self._spawn(i)
                        continue
                    reward[i] += r
                    lap[i] |= lp
                    if reason:
                        done[i] = True
                        term[i] = reason not in (TermReason.TIME_LIMIT, TermReason.FINISHED)
                        trunc[i] = not term[i]
                        self._obs(i, final_obs)
                        self._info(i, info, bool(lap[i]))
                        self._spawn(i)
        for i in range(n):
            self._obs(i, obs)
            if not done[i]:
                self._info(i, info, bool(lap[i]))
        return encode_state(obs, reward.astype(np.float32), term, trunc, final_obs, info, self._state_buf)

    def run(self) -> None:
        try:
            self._run()
        except BaseException as e:  # noqa: BLE001 - surfaced to tests via self.error
            self.error = e

    def _run(self) -> None:
        sock = None
        for _ in range(200):
            try:
                sock = socket.create_connection((self.host, self.port), timeout=5.0)
                break
            except OSError:
                time.sleep(0.05)
        if sock is None:
            raise ConnectionError(f"mock_unity: cannot connect to {self.port}")
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(60.0)
        with sock:
            hello = {"protocol": PROTOCOL, "env": "mock_circle", "unity": "mock", "build_id": "mock", "num_agents": self.n,
                     "obs_dim": OBS_DIM, "act_dim": ACT_DIM, "obs_layout_hash": self.obs_hash,
                     "env_config_hash": self.env_hash, "fixed_dt": self.dt, "decision_period": self.k,
                     "max_episode_decisions": self.max_dec, "info_struct": INFO_STRUCT,
                     "track_id": self.track_id, "track_index": self.track_index, "track_length_m": LENGTH,
                     "track_checkpoints": CHECKPOINTS, "track_half_width": HALF_WIDTH, "track_hash": self.track_hash}
            _send(sock, MSG_HELLO, 0, json.dumps(hello).encode())
            t, seq, payload = _recv(sock)
            if t != MSG_CONFIG:
                return
            cfg = json.loads(payload)
            # Same order as BridgeProtocol.CheckConfig: the track id first, then the hashes.
            code = None
            if cfg.get("expected_track_id") and cfg["expected_track_id"] != self.track_id:
                code = ERR_TRACK_MISMATCH
            elif not ((not cfg.get("expected_env_config_hash") or cfg["expected_env_config_hash"] == self.env_hash) and
                      (not cfg.get("expected_obs_layout_hash") or cfg["expected_obs_layout_hash"] == self.obs_hash)):
                code = ERR_HASH_MISMATCH
            if code and cfg.get("strict", True):
                _send(sock, MSG_ERROR, seq, json.dumps({"code": code, "message": "mock", "fatal": True}).encode())
                return
            _send(sock, MSG_READY, seq, b'{"ok":true}')
            self.reset_all(0, 0, 0)
            while True:
                try:
                    t, seq, payload = _recv(sock)
                except ConnectionError:
                    return
                self.requests += 1
                if t == MSG_CLOSE:
                    return
                if t == MSG_RESET:
                    self.reset_all(*RESET.unpack(payload))
                    _send(sock, MSG_STATE, seq, self.state_payload(None))
                elif t == MSG_STEP:
                    a = np.frombuffer(payload, np.float32).reshape(self.n, ACT_DIM)
                    if not np.isfinite(a).all():
                        _send(sock, MSG_ERROR, seq, b'{"code":"BAD_ACTION","message":"non-finite","fatal":false}')
                        continue
                    self.steps += 1
                    if self.crash_at is not None and self.steps == self.crash_at:
                        return
                    if self.hang_at is not None and self.steps == self.hang_at:
                        time.sleep(self.hang_s)
                    echo = seq + 1 if self.desync_at is not None and self.steps == self.desync_at else seq
                    _send(sock, MSG_STATE, echo, self.state_payload(a))
                else:
                    _send(sock, MSG_ERROR, seq, b'{"code":"UNEXPECTED_MSG","message":"mock","fatal":false}')

    def start_thread(self) -> threading.Thread:
        th = threading.Thread(target=self.run, name=f"mock_unity:{self.port}", daemon=True)
        th.start()
        return th


def _send(sock: socket.socket, t: int, seq: int, payload: bytes | bytearray) -> None:
    sock.sendall(HEADER.pack(MAGIC, VERSION, t, seq, len(payload)) + bytes(payload))


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return bytes(buf)


def _recv(sock: socket.socket) -> tuple[int, int, bytes]:
    magic, version, t, seq, n = HEADER.unpack(_recv_exactly(sock, HEADER.size))
    if magic != MAGIC or version != VERSION:
        raise ConnectionError("bad header")
    return t, seq, _recv_exactly(sock, n) if n else b""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=6005)
    p.add_argument("--num-agents", type=int, default=4)
    p.add_argument("--max-episode-decisions", type=int, default=3000)
    p.add_argument("--track-id", default="mock_circle")
    p.add_argument("--track-index", type=int, default=-1)
    a = p.parse_args()
    MockUnity(a.port, a.num_agents, max_episode_decisions=a.max_episode_decisions, track_id=a.track_id,
              track_index=a.track_index).run()


if __name__ == "__main__":
    main()
