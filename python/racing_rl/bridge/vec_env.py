"""UnityVecEnv: one Unity process with N agents behind the M3 TCP bridge (gymnasium VectorEnv, SAME_STEP autoreset)."""

from __future__ import annotations

import json
import os
import socket
import time
import warnings
import weakref
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium.vector.utils import batch_space

from .. import paths
from .errors import (BridgeError, DesyncError, ProtocolMismatchError, RemoteError, UnityCrashedError,
                     UnityLaunchError, UnityTimeoutError)
from .protocol import (ACT_DIM, FROZEN_ENV_CONFIG_HASH, HELLO_TRACK_FIELDS, INFO_FIELDS, INFO_STRUCT, MSG_CLOSE,
                       MSG_CONFIG, MSG_ERROR, MSG_HELLO, MSG_NAMES, MSG_READY, MSG_RESET, MSG_STATE, MSG_STEP, OBS_DIM,
                       OBS_HIGH, OBS_LOW, PROTOCOL, RESET, StateView, decode_state, layout_hash)
from .tracks import TrackSpec, load_catalog, resolve_track
from .transport import FramedSocket, bind_server
from .unity_process import UnityProcess

EDITOR_STEP_TIMEOUT_S = 600.0
MAX_RESTARTS_PER_HOUR = 3
# expected_env_hash default: the track's catalog hash (Track_A without a track, as in M3-M5; None for an unfrozen proc seed)
AUTO = "auto"
TRACK_FLAGS = ("-trackName", "-trackIndex")


class _Resources:
    """Socket + process, owned separately so weakref.finalize can clean up without keeping the env alive."""

    def __init__(self):
        self.sock: FramedSocket | None = None
        self.proc: UnityProcess | None = None
        self.seq = 0

    def shutdown(self, graceful: bool = True) -> None:
        if self.sock is not None:
            if graceful:
                try:
                    self.seq += 1
                    self.sock.send(MSG_CLOSE, self.seq)
                except BridgeError:
                    pass
            self.sock.close()
            self.sock = None
        if self.proc is not None:
            if not (graceful and self.proc.wait(5.0)):
                self.proc.kill_tree()
            self.proc = None


class UnityVecEnv(gymnasium.vector.VectorEnv):
    """
    N agents of one Unity process. Python is the TCP server; Unity (player or Editor in Play mode) connects.

    step() returns (obs, reward, terminated, truncated, infos). Done agents are reset inside the same STEP
    (SAME_STEP): obs is the new episode's first observation, infos["final_obs"][i] / infos["final_info"]
    hold the finished episode (mask "_final_obs"). infos[field] carries every RACE_INFO_V1 field; for a done
    agent it describes the finished episode. exe_path=None waits for the Unity Editor (Play) instead.

    track (M6): catalog id, asset name or proc:<seed> (racing_rl.bridge.tracks). The player gets -trackName, CONFIG
    carries expected_track_id and HELLO's track_id / track_index / track_hash are checked; the track is fixed for the
    process (C0.20). In Editor mode (exe_path=None) the scene's BridgeDriver.trackOverride selects the track and the
    CONFIG check guards it. expected_env_hash=AUTO takes the track's catalog hash. track=None is the M5 path: no flag,
    no expected_track_id, Track_A's hash. infos["track_index"] is HELLO's track_index (-1 for proc) when the build
    reports it; env.track_info holds the HELLO track fields (None for a pre-M6 build).
    """

    metadata = {"autoreset_mode": gymnasium.vector.AutoresetMode.SAME_STEP}

    def __init__(self, exe_path: str | os.PathLike | None, num_agents: int = 16, port: int = 6005,
                 expected_env_hash: str | None = AUTO, log_dir: str | os.PathLike = paths.RUNS / "unity_logs",
                 step_timeout_s: float | None = None, launch_timeout_s: float = 180.0, *, host: str = "127.0.0.1",
                 strict: bool = True, extra_args: list[str] | None = None, headless: bool = True,
                 record_rtt: bool = False, timing_log: str | os.PathLike | None = None, launch_retries: int = 1,
                 track: str | None = None):
        self.exe_path = None if exe_path is None else str(Path(exe_path).resolve())
        if self.exe_path is not None and not Path(self.exe_path).is_file():
            raise UnityLaunchError(f"Unity player not found: {self.exe_path}")
        self.track: TrackSpec | None = None if track is None else resolve_track(track)
        if self.track is not None and any(f in TRACK_FLAGS for f in (extra_args or [])):
            raise ValueError("pass the track either as track= or as -trackName/-trackIndex in extra_args, not both")
        if expected_env_hash == AUTO:
            expected_env_hash = FROZEN_ENV_CONFIG_HASH if self.track is None else self.track.expected_env_hash
        self.requested_agents = num_agents
        self.host = host
        self.base_port = port
        self.expected_env_hash = expected_env_hash
        self.expected_obs_hash = layout_hash()
        self.strict = strict
        self.log_dir = Path(log_dir)
        self.step_timeout_s = step_timeout_s if step_timeout_s is not None else (
            60.0 if self.exe_path else EDITOR_STEP_TIMEOUT_S)
        self.launch_timeout_s = launch_timeout_s
        self.extra_args = list(extra_args or [])
        if self.track is not None and self.exe_path:
            self.extra_args += ["-trackName", self.track.id]
        self.timing_log = None if timing_log is None else Path(timing_log)
        if self.timing_log is not None:
            self.extra_args += ["-bridgeTimingLog", str(self.timing_log.resolve())]
        self.headless = headless
        self.launch_retries = launch_retries
        self.record_rtt = record_rtt
        self.rtt_seq: list[int] = []
        self.rtt_ns: list[int] = []

        self._res = _Resources()
        self._finalizer = weakref.finalize(self, self._res.shutdown, True)
        self.hello: dict = {}
        self.track_info: dict | None = None
        self.port: int | None = None
        self.nonfinite_obs = 0
        self.requests_sent = 0
        self.replies_received = 0
        self._restarts: list[float] = []
        self._pending = False
        self._last_seed: int | None = None
        self._start_mode = 0
        self._max_laps = 0

        self._launch()
        n = int(self.hello["num_agents"])
        self.num_envs = n
        self.single_observation_space = gymnasium.spaces.Box(OBS_LOW, OBS_HIGH, (OBS_DIM,), np.float32)
        self.single_action_space = gymnasium.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, n)
        self.action_space = batch_space(self.single_action_space, n)
        self._actions = np.zeros((n, ACT_DIM), np.float32)
        self._track_index = None  # shared, read-only per-agent array for infos["track_index"]
        if self.track_info is not None:
            self._track_index = np.full(n, self.track_info["track_index"], np.int32)
            self._track_index.flags.writeable = False
        self.closed = False

    # ------------------------------------------------------------------ launch / handshake
    @property
    def process(self) -> UnityProcess | None:
        return self._res.proc

    def _launch(self) -> None:
        port = self.base_port
        for attempt in range(self.launch_retries + 1):
            srv, bound = bind_server(self.host, port)
            self.port = bound
            proc = None
            try:
                if self.exe_path:
                    log = self.log_dir / f"unity_{bound}_{int(time.time() * 1000)}.log"
                    proc = UnityProcess(self.exe_path, bound, self.requested_agents, log, self.extra_args, self.headless)
                    try:
                        proc.start()
                    except OSError as e:
                        raise UnityLaunchError(f"cannot start {self.exe_path}: {e}") from e
                self._res.proc = proc
                conn = self._accept(srv, proc)
                self._res.sock = FramedSocket(conn, self.launch_timeout_s)
                self._handshake()
                self._res.sock.set_timeout(self.step_timeout_s)
                return
            except UnityLaunchError as e:
                self._res.shutdown(graceful=False)
                if attempt >= self.launch_retries:
                    raise
                warnings.warn(f"Unity launch failed on port {bound}, retrying on another port: {e}")
                port = bound + 1
            except BaseException:
                self._res.shutdown(graceful=False)
                raise
            finally:
                srv.close()

    def _accept(self, srv: socket.socket, proc: UnityProcess | None) -> socket.socket:
        deadline = time.monotonic() + self.launch_timeout_s
        srv.settimeout(0.5)
        while True:
            try:
                conn, _ = srv.accept()
                return conn
            except socket.timeout:
                pass
            if proc is not None and not proc.alive():
                raise UnityLaunchError(f"Unity exited with code {proc.returncode()} before connecting.\n{proc.tail_log()}")
            if time.monotonic() > deadline:
                tail = proc.tail_log() if proc else "(Editor mode: press Play in Race_Bridge.unity)"
                raise UnityLaunchError(f"Unity did not connect to port {self.port} within {self.launch_timeout_s:.0f}s\n{tail}")

    def _handshake(self) -> None:
        sock = self._res.sock
        t, seq, payload = sock.recv()
        if t == MSG_ERROR:
            self._raise_remote(payload)
        if t != MSG_HELLO:
            raise DesyncError(f"expected HELLO, got {MSG_NAMES.get(t, hex(t))}")
        self.hello = hello = json.loads(bytes(payload).decode("utf-8"))
        problems = []
        if hello.get("protocol") != PROTOCOL:
            problems.append(f"protocol {hello.get('protocol')} != {PROTOCOL}")
        if hello.get("obs_dim") != OBS_DIM or hello.get("act_dim") != ACT_DIM:
            problems.append(f"obs/act dim {hello.get('obs_dim')}/{hello.get('act_dim')} != {OBS_DIM}/{ACT_DIM}")
        if hello.get("info_struct") != INFO_STRUCT:
            problems.append(f"info_struct {hello.get('info_struct')} != {INFO_STRUCT}")
        if hello.get("obs_layout_hash") != self.expected_obs_hash:
            problems.append(f"obs_layout_hash {hello.get('obs_layout_hash')} != {self.expected_obs_hash}")
        if self.expected_env_hash and hello.get("env_config_hash") != self.expected_env_hash:
            problems.append(f"env_config_hash {hello.get('env_config_hash')} != {self.expected_env_hash}")
        if self.exe_path and hello.get("num_agents") != self.requested_agents:
            problems.append(f"num_agents {hello.get('num_agents')} != {self.requested_agents}")
        problems += self._track_problems(hello)
        if problems and self.strict:
            self._res.shutdown(graceful=True)
            raise ProtocolMismatchError("; ".join(problems))
        for p in problems:
            warnings.warn("bridge handshake (strict=False): " + p)
        self.track_info = _track_info(hello)

        self._res.seq = 1
        cfg = {"expected_obs_layout_hash": self.expected_obs_hash,
               "expected_env_config_hash": self.expected_env_hash or "", "strict": self.strict}
        if self.track is not None:
            cfg["expected_track_id"] = self.track.id
        sock.send(MSG_CONFIG, 1, json.dumps(cfg).encode("utf-8"))
        t, seq, payload = sock.recv()
        if t == MSG_ERROR:
            err = json.loads(bytes(payload).decode("utf-8"))
            self._res.shutdown(graceful=False)
            raise ProtocolMismatchError(f"Unity rejected CONFIG: {err}")
        if t != MSG_READY or seq != 1:
            raise DesyncError(f"expected READY seq=1, got {MSG_NAMES.get(t, hex(t))} seq={seq}")

    def _track_problems(self, hello: dict) -> list[str]:
        if self.track is None:
            return []
        if "track_id" not in hello:
            return ["HELLO has no track fields: the build predates M6 step 4 (track selection needs outputs/builds/RaceEnv)"]
        if hello["track_id"] != self.track.id:
            hint = "" if self.exe_path else " (Editor: set BridgeDriver.trackOverride in the scene)"
            return [f"track_id {hello['track_id']} != {self.track.id}{hint}"]
        out = []
        if hello.get("track_index") != self.track.index:
            out.append(f"track_index {hello.get('track_index')} != {self.track.index}")
        e = self.track.entry
        if e is not None and hello.get("track_hash") != e.track_hash:
            out.append(f"track_hash {hello.get('track_hash')} != {e.track_hash}")
        return out

    # ------------------------------------------------------------------ request / reply
    def _send(self, msg_type: int, payload) -> None:
        if self._res.sock is None:
            raise UnityCrashedError("bridge is not connected (closed or crashed)")
        if self._pending:
            raise DesyncError("request already in flight (strict request/response, no pipelining)")
        self._res.seq += 1
        self._t_send = time.perf_counter_ns()
        try:
            self._res.sock.send(msg_type, self._res.seq, payload)
        except BridgeError:
            self._fail()
            raise
        self._pending = True
        self.requests_sent += 1

    def _recv_state(self) -> StateView:
        if not self._pending:
            raise DesyncError("step_wait/recv without a pending request")
        try:
            t, seq, payload = self._res.sock.recv()
        except UnityTimeoutError as e:
            self._fail()
            raise UnityTimeoutError(f"no reply within {self.step_timeout_s:.0f}s (seq {self._res.seq}); Unity killed") from e
        except BridgeError:
            self._fail()
            raise
        self._pending = False
        self.replies_received += 1
        if self.record_rtt:
            self.rtt_seq.append(seq)
            self.rtt_ns.append(time.perf_counter_ns() - self._t_send)
        if t == MSG_ERROR:
            self._raise_remote(payload)
        if t != MSG_STATE or seq != self._res.seq:
            self._fail()
            raise DesyncError(f"expected STATE seq={self._res.seq}, got {MSG_NAMES.get(t, hex(t))} seq={seq}")
        return decode_state(payload, self.num_envs)

    def _raise_remote(self, payload) -> None:
        try:
            err = json.loads(bytes(payload).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            err = {"code": "UNKNOWN", "message": bytes(payload).hex(), "fatal": True}
        self._pending = False
        if err.get("fatal", True):
            self._fail()
        raise RemoteError(err.get("code", "UNKNOWN"), err.get("message", ""), bool(err.get("fatal", True)))

    def _fail(self) -> None:
        self._pending = False
        self._res.shutdown(graceful=False)

    # ------------------------------------------------------------------ gymnasium API
    def reset(self, *, seed: int | list[int] | None = None, options: dict | None = None):
        super().reset(seed=None if seed is None else int(np.atleast_1d(seed)[0]))
        if seed is None:
            seed = int(self.np_random.integers(0, 2**31 - 1))
        self._send_reset(int(np.atleast_1d(seed)[0]), options)
        return self._finish_reset()

    def _send_reset(self, seed: int, options: dict | None) -> None:
        """RESET: agent i is seeded with seed + i; start_mode/max_laps persist until the next RESET."""
        options = options or {}
        self._start_mode = int(options.get("start_mode", self._start_mode))
        self._max_laps = int(options.get("max_laps", self._max_laps))
        self._last_seed = seed
        self._send(MSG_RESET, RESET.pack(seed, self._start_mode, self._max_laps))

    def _finish_reset(self):
        sv = self._recv_state()
        obs = self._check_obs(sv.obs.copy())
        return obs, self._infos(sv, reset=True)

    def step_async(self, actions) -> None:
        a = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, ACT_DIM)
        if not np.isfinite(a).all():
            raise ValueError("non-finite action (C0.5)")
        np.copyto(self._actions, a)
        self._send(MSG_STEP, memoryview(self._actions).cast("B"))

    def step_wait(self):
        sv = self._recv_state()
        obs = self._check_obs(sv.obs.copy())
        reward = sv.reward.copy()
        terminated = sv.terminated.astype(bool)
        truncated = sv.truncated.astype(bool)
        return obs, reward, terminated, truncated, self._infos(sv, reset=False)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def _check_obs(self, obs: np.ndarray) -> np.ndarray:
        if not np.isfinite(obs).all():
            bad = int((~np.isfinite(obs)).any(axis=1).sum())
            self.nonfinite_obs += bad
            warnings.warn(f"{bad} agent(s) returned non-finite observations (total {self.nonfinite_obs})")
        return obs

    def _infos(self, sv: StateView, reset: bool) -> dict:
        n = self.num_envs
        ones = np.ones(n, bool)
        infos: dict = {}
        info = sv.info.copy()
        for f in INFO_FIELDS:
            infos[f] = info[f]
            infos["_" + f] = ones
        if self._track_index is not None:
            infos["track_index"] = self._track_index
            infos["_track_index"] = ones
        if not reset:
            done = (sv.terminated | sv.truncated).astype(bool)
            infos["final_obs"] = sv.final_obs.copy()
            infos["_final_obs"] = done
            infos["final_info"] = {f: info[f] for f in INFO_FIELDS}
            infos["_final_info"] = done
        return infos

    # ------------------------------------------------------------------ lifecycle
    def restart(self) -> tuple[np.ndarray, dict]:
        """Relaunch after a crash/timeout (max 3/hour) and reset with the last seed/options."""
        now = time.monotonic()
        self._restarts = [t for t in self._restarts if now - t < 3600.0]
        if len(self._restarts) >= MAX_RESTARTS_PER_HOUR:
            raise UnityCrashedError(f"more than {MAX_RESTARTS_PER_HOUR} restarts within an hour")
        self._restarts.append(now)
        self._res.shutdown(graceful=False)
        self._pending = False
        self._launch()
        seed = self._last_seed
        return self.reset(seed=seed, options={"start_mode": self._start_mode, "max_laps": self._max_laps})

    def close_extras(self, **kwargs) -> None:
        self._pending = False
        self._res.shutdown(graceful=True)

    @property
    def connected(self) -> bool:
        return self._res.sock is not None


def _track_info(hello: dict) -> dict | None:
    """HELLO track fields + env_config_hash and the catalog profile; None when the build predates M6 step 4."""
    if "track_id" not in hello:
        return None
    info = {f: hello.get(f) for f in HELLO_TRACK_FIELDS}
    info["env_config_hash"] = hello.get("env_config_hash")
    entry = load_catalog().entry(info["track_id"])
    info["profile"] = entry.profile if entry else ("Procedural" if info["track_index"] == -1 else None)
    return info
