"""
Watch a trained policy drive live in the Unity Editor, Play mode. Scenes/Race_Watch.unity has a single car
(menu Racing/Setup Watch Scene); Scenes/Race_Bridge.unity runs the 16-agent training setup (cars do not collide).

Start this first (Python is the TCP server), then press Play within --wait seconds. Decisions are throttled to
real time (dt = fixed_dt · decision_period = 0.1 s at --speed 1); without the throttle the Editor would process as many
requests as fit its 50 ms frame budget and the cars would run many times faster than real time. Ctrl+C stops.

    python -m racing_rl.train.watch
    python -m racing_rl.train.watch --policy torch:../outputs/runs/m5/parity_s1/best.pt
    python -m racing_rl.train.watch --policy onnx:../outputs/benchmarks/models/mlagents_baseline_s1.onnx

Default policy: outputs/benchmarks/models/custom_ppo_s2.pt (best custom PPO, flying lap 38.68 s). EvalGrid, 3 laps,
deterministic μ; agents auto-reset after the race and keep driving. The Editor renders roughly once per decision.

--track (M6): the Editor cannot take command-line flags, so first type the same id into the scene's
BridgeDriver.trackOverride field (catalog id or proc:<seed>); --track then expects that track (CONFIG
expected_track_id and its catalog hash) and refuses any other one:

    python -m racing_rl.train.watch --track Track_D
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from racing_rl import paths
from racing_rl.bridge.protocol import StartMode
from racing_rl.bridge.vec_env import UnityVecEnv
from racing_rl.train.evaluate import load_policy

DEFAULT_POLICY = f"torch:{paths.MODELS / 'custom_ppo_s2.pt'}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default=DEFAULT_POLICY, help="torch:<ckpt> | onnx:<path>")
    ap.add_argument("--port", type=int, default=6005)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--speed", type=float, default=1.0, help="sim time / wall time (0 = unthrottled)")
    ap.add_argument("--wait", type=float, default=300.0, help="seconds to wait for Play")
    ap.add_argument("--track", help="track set in BridgeDriver.trackOverride (catalog id or proc:<seed>)")
    a = ap.parse_args()

    policy, _, model = load_policy(a.policy)
    where = f" with BridgeDriver.trackOverride = {a.track}" if a.track else ""
    print(f"{model} loaded. Listening on 127.0.0.1:{a.port}: press Play in Race_Watch.unity (or Race_Bridge.unity){where} "
          f"now (timeout {a.wait:.0f} s)...", flush=True)
    env = UnityVecEnv(None, port=a.port, launch_timeout_s=a.wait, track=a.track)
    try:
        dt = env.hello["fixed_dt"] * env.hello["decision_period"]
        period = dt / a.speed if a.speed > 0 else 0.0
        track = f", track {env.track_info['track_id']}" if env.track_info else ""
        print(f"Connected: {env.num_envs} agents{track}, {dt:.2f} s/decision. Camera follows agent 0. Ctrl+C to stop.",
              flush=True)
        obs, _ = env.reset(seed=a.seed, options={"start_mode": int(StartMode.EVAL_GRID), "max_laps": a.laps})
        best = float("inf")
        while True:
            t0 = time.perf_counter()
            obs, _, term, trunc, info = env.step(np.asarray(policy(obs), np.float32))
            if info["lap_completed"][0]:
                lap = float(info["last_lap_s"][0])
                best = min(best, lap)
                print(f"agent 0 lap {int(info['laps'][0])}: {lap:.2f} s (best {best:.2f} s)", flush=True)
            if term[0] or trunc[0]:
                print("agent 0 episode ended -> reset", flush=True)
            if period:
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
