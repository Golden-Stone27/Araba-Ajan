"""MultiUnityVecEnv: K Unity processes × N agents. step_async sends to all, step_wait collects from all."""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import gymnasium
import numpy as np
from gymnasium.vector.utils import batch_space

from .tracks import BENCHMARK_ID, composite_hash, resolve_track
from .vec_env import AUTO, UnityVecEnv


class MultiUnityVecEnv(gymnasium.vector.VectorEnv):
    """
    Agent j of process k is env index k·N + j. Process k is seeded with seed + k·N (distinct spawn streams).

    tracks (M6): process k runs tracks[k % len(tracks)] for its whole life (C0.20: no track change on RESET) and is
    checked against that track's own catalog hash. track_ids lists each process's track, env_hashes maps every
    distinct track id to its HELLO env_config_hash and env_config_hash is their composite (tracks.composite_hash; the
    plain hash for a single track). tracks=None is the M5 setup (Track_A everywhere, no track flag).
    """

    metadata = {"autoreset_mode": gymnasium.vector.AutoresetMode.SAME_STEP}

    def __init__(self, exe_path: str | os.PathLike | None, num_processes: int = 2, num_agents: int = 16,
                 base_port: int = 6005, expected_env_hash: str | None = AUTO, log_dir: str | os.PathLike = "runs/unity_logs",
                 step_timeout_s: float | None = None, launch_timeout_s: float = 180.0, *,
                 tracks: Sequence[str] | None = None, **kwargs):
        ports = [base_port + 25 * k for k in range(num_processes)]  # bind_server scans +20; keep ranges disjoint
        per_proc: list[str | None] = [None] * num_processes
        if tracks:
            ids = [resolve_track(t).id for t in tracks]  # unknown names fail before any process starts
            if len(ids) > num_processes:
                raise ValueError(f"{len(ids)} tracks need at least as many processes (got {num_processes}); "
                                 "the track is fixed per process")
            if len(set(ids)) > 1 and expected_env_hash not in (AUTO, None):
                raise ValueError("expected_env_hash names one environment; with several tracks every process is "
                                 "checked against its own catalog hash (leave it AUTO)")
            if num_processes % len(ids):
                warnings.warn(f"{num_processes} processes over {len(ids)} tracks: unequal agent share per track")
            per_proc = [ids[k % len(ids)] for k in range(num_processes)]

        def make(port: int, track: str | None) -> UnityVecEnv:
            return UnityVecEnv(exe_path, num_agents, port, expected_env_hash, log_dir, step_timeout_s, launch_timeout_s,
                               track=track, **kwargs)

        with ThreadPoolExecutor(max_workers=num_processes) as pool:
            futures = [pool.submit(make, p, t) for p, t in zip(ports, per_proc)]
            envs, error = [], None
            for f in futures:
                try:
                    envs.append(f.result())
                except BaseException as e:  # noqa: BLE001 - close the ones that did start, then re-raise
                    error = error or e
        if error is not None:
            for e in envs:
                e.close()
            raise error

        self.envs = envs
        self.track_ids = [e.track_info["track_id"] if e.track_info else BENCHMARK_ID for e in envs]
        self.env_hashes: dict[str, str] = {}
        for tid, e in zip(self.track_ids, envs):
            self.env_hashes.setdefault(tid, e.hello["env_config_hash"])
        self.num_agents = num_agents
        self.num_envs = sum(e.num_envs for e in envs)
        self.single_observation_space = envs[0].single_observation_space
        self.single_action_space = envs[0].single_action_space
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self._slices = []
        start = 0
        for e in envs:
            self._slices.append(slice(start, start + e.num_envs))
            start += e.num_envs
        self.closed = False

    @property
    def env_config_hash(self) -> str:
        return composite_hash(self.env_hashes)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**31 - 1))
        for e, s in zip(self.envs, self._slices):
            e._send_reset(seed + s.start, options)
        results = [e._finish_reset() for e in self.envs]
        obs = np.concatenate([r[0] for r in results])
        return obs, _merge_infos([r[1] for r in results])

    def step_async(self, actions) -> None:
        a = np.asarray(actions, np.float32)
        for e, s in zip(self.envs, self._slices):
            e.step_async(a[s])

    def step_wait(self):
        parts = [e.step_wait() for e in self.envs]
        obs = np.concatenate([p[0] for p in parts])
        rew = np.concatenate([p[1] for p in parts])
        term = np.concatenate([p[2] for p in parts])
        trunc = np.concatenate([p[3] for p in parts])
        return obs, rew, term, trunc, _merge_infos([p[4] for p in parts])

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close_extras(self, **kwargs) -> None:
        for e in self.envs:
            e.close()


def _merge_infos(infos: list[dict]) -> dict:
    out: dict = {}
    for k in infos[0]:
        v = infos[0][k]
        if isinstance(v, dict):
            out[k] = _merge_infos([i[k] for i in infos])
        else:
            out[k] = np.concatenate([i[k] for i in infos])
    return out
