"""
M6 step 4 checks against the real bridge build.

    python scripts/m6_bridge_check.py smoke
    python scripts/m6_bridge_check.py regress [--all-variants]
    python scripts/m6_bridge_check.py determinism [--tracks Track_D Track_A] [--steps 1500]

smoke        HELLO of every catalog track and proc:7 (env_config_hash and track fields vs C0.20), the UNKNOWN_TRACK path
             (unknown name, out-of-range index, conflicting flags) and CONFIG expected_track_id (TRACK_MISMATCH).
regress      Track_A C0.10 evaluation of the six M5 models with the new build. Every per-seed metric and every trace array
             must be bit-identical to benchmarks/{custom_ppo,mlagents_bridge}.json and benchmarks/eval/traces/*.npz.
             Variants: no track flag (the M5 command line) and -trackName Track_A for all six, -trackIndex 0 for custom s1
             (--all-variants: every variant for every model).
determinism  Per track and scenario, two fresh processes with -trackName T: RESET(seed) + N STEPs, twice per process.
             Scenarios (TrainRandom spawns over the whole track): a fixed action sequence, and the M5 policy (deterministic μ).
             The SHA-256 of every STATE (obs, reward, flags, final_obs, RACE_INFO) must match across the processes and
             across the two RESETs of one process (RESET rebuilds the cars, contracts C0.17).
Per-run outputs go to runs/m6/ (gitignored); summaries to benchmarks/eval/m6_*.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from racing_rl.bridge import RemoteError, UnityVecEnv  # noqa: E402
from racing_rl.bridge.protocol import (ERR_TRACK_MISMATCH, ERR_UNKNOWN_TRACK, HELLO_TRACK_FIELDS, INFO_FIELDS,  # noqa: E402
                                       MSG_CLOSE, MSG_CONFIG, MSG_ERROR, MSG_HELLO, MSG_READY, layout_hash)
from racing_rl.bridge.transport import FramedSocket, bind_server  # noqa: E402
from racing_rl.bridge.unity_process import UnityProcess  # noqa: E402

EXE = REPO / "Builds" / "RaceEnv" / "RaceEnv.exe"
OUT = REPO / "runs" / "m6"
LOGS = OUT / "unity_logs"

# contracts C0.20 (env_config_hash) and BridgeTrackTests (track_hash = the track part alone)
FROZEN = {
    "Track_A": (0, "90240ee2b1a58b5b", "0f2fbf3481a1a24f", 114),
    "Track_B": (1, "038a104393cbfb72", "6ce4f5d0b0d07eda", 111),
    "Track_C": (2, "0e099647315ed638", "ff0ea58517167d6f", 126),
    "Track_D": (3, "dbce4b7f772fc7b4", "04f6036ec025016a", 109),
    "proc:7": (-1, "59966c9f08fc1027", None, None),
}


def hello_of(flags: list[str], expected_env_hash: str | None, port: int) -> dict:
    env = UnityVecEnv(EXE, num_agents=2, port=port, expected_env_hash=expected_env_hash, log_dir=LOGS, extra_args=flags)
    try:
        env.reset(seed=1000)
        env.step(np.zeros((2, 2), np.float32))
        return dict(env.hello)
    finally:
        env.close()


def raw_config(flags: list[str], cfg: dict, port: int) -> tuple[dict, int, dict]:
    """HELLO/CONFIG by hand (UnityVecEnv sends expected_track_id from step 5 on). Returns (hello, reply type, reply)."""
    srv, bound = bind_server("127.0.0.1", port)
    proc = UnityProcess(EXE, bound, 2, LOGS / f"raw_{bound}_{int(time.time() * 1000)}.log", flags)
    try:
        proc.start()
        srv.settimeout(180.0)
        conn, _ = srv.accept()
        sock = FramedSocket(conn, 60.0)
        t, _, payload = sock.recv()
        assert t == MSG_HELLO, t
        hello = json.loads(bytes(payload))
        sock.send(MSG_CONFIG, 1, json.dumps(cfg).encode())
        t, _, payload = sock.recv()
        reply = json.loads(bytes(payload))
        if t == MSG_READY:
            sock.send(MSG_CLOSE, 2)
        sock.close()
        proc.wait(30.0)
        return hello, t, reply
    finally:
        srv.close()
        if proc.alive():
            proc.kill_tree()


def smoke(a) -> dict:
    port = a.port
    rows, ok = [], True
    cases = [([], "Track_A"), (["-trackName", "Track_A"], "Track_A"), (["-trackIndex", "0"], "Track_A"),
             (["-trackName", "Track_B"], "Track_B"), (["-trackName", "TrackDefinition_B"], "Track_B"),
             (["-trackIndex", "2"], "Track_C"), (["-trackName", "Track_D", "-trackIndex", "3"], "Track_D"),
             (["-trackName", "proc:7"], "proc:7")]
    for flags, tid in cases:
        index, env_hash, track_hash, gates = FROZEN[tid]
        h = hello_of(flags, env_hash, port)
        port += 1
        good = (h["track_id"] == tid and h["track_index"] == index and h["env_config_hash"] == env_hash
                and set(HELLO_TRACK_FIELDS) <= set(h) and h["protocol"] == 1
                and (track_hash is None or h["track_hash"] == track_hash) and (gates is None or h["track_checkpoints"] == gates))
        ok &= good
        rows.append({"flags": flags, "ok": good, **{k: h[k] for k in ("env", "env_config_hash", *HELLO_TRACK_FIELDS)}})
        print(f"[smoke] {' '.join(flags) or '(no flag)':40s} -> {h['track_id']} idx {h['track_index']} "
              f"env {h['env_config_hash']} L {h['track_length_m']:.1f} gates {h['track_checkpoints']} {'OK' if good else 'FAIL'}")

    for flags in (["-trackName", "Nope"], ["-trackIndex", "9"], ["-trackName", "Track_B", "-trackIndex", "2"],
                  ["-trackName", "proc:7", "-trackIndex", "0"], ["-trackIndex", "x"]):
        try:
            hello_of(flags, None, port)
            code, msg = None, "connected"
        except RemoteError as e:
            code, msg = e.code, str(e)
        port += 1
        good = code == ERR_UNKNOWN_TRACK
        ok &= good
        rows.append({"flags": flags, "ok": good, "error": code, "message": msg})
        print(f"[smoke] {' '.join(flags):40s} -> {msg} {'OK' if good else 'FAIL'}")

    base = {"expected_obs_layout_hash": layout_hash(), "expected_env_config_hash": FROZEN["Track_D"][1]}
    for expected, strict, want in (("Track_D", True, MSG_READY), ("Track_A", True, MSG_ERROR), ("Track_A", False, MSG_READY)):
        hello, t, reply = raw_config(["-trackName", "Track_D"], {**base, "expected_track_id": expected, "strict": strict}, port)
        port += 1
        good = t == want and (t != MSG_ERROR or (reply.get("code") == ERR_TRACK_MISMATCH and reply.get("fatal") is True))
        ok &= good
        rows.append({"flags": ["-trackName", "Track_D"], "expected_track_id": expected, "strict": strict, "ok": good,
                     "reply": "READY" if t == MSG_READY else reply})
        print(f"[smoke] CONFIG expected_track_id={expected} strict={strict} -> "
              f"{'READY' if t == MSG_READY else reply} {'OK' if good else 'FAIL'}")
    return {"check": "smoke", "ok": ok, "cases": rows}


# ---------------------------------------------------------------------------------------------------- regression
MODELS = [("custom", s, f"torch:{REPO / 'benchmarks' / 'models' / f'custom_ppo_s{s}.pt'}", "custom_ppo.json") for s in (1, 2, 3)] + \
         [("mlagents", s, f"onnx:{REPO / 'benchmarks' / 'models' / f'mlagents_baseline_s{s}.onnx'}", "mlagents_bridge.json")
          for s in (1, 2, 3)]
SKIP_KEYS = {"wallclock_s", "trace", "policy_spec", "checkpoint_step"}  # run metadata, not results


def same(x, y) -> bool:
    if isinstance(x, float) and isinstance(y, float):
        return x.hex() == y.hex() or (np.isnan(x) and np.isnan(y))
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return len(x) == len(y) and all(same(u, v) for u, v in zip(x, y))
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    return x == y


def compare_traces(ref: Path, new: Path) -> list[str]:
    r, n = np.load(ref), np.load(new)
    diff = sorted(set(r.files) ^ set(n.files))
    for k in sorted(set(r.files) & set(n.files)):
        if r[k].dtype != n[k].dtype or r[k].shape != n[k].shape or r[k].tobytes() != n[k].tobytes():
            diff.append(k)
    return diff


def regress(a) -> dict:
    from racing_rl.train.evaluate import evaluate
    variants = {"none": [], "name": ["-trackName", "Track_A"], "index": ["-trackIndex", "0"]}
    port, rows, ok = a.port, [], True
    for kind, seed, spec, ref_file in MODELS:
        ref = {p["seed"]: p for p in json.loads((REPO / "benchmarks" / ref_file).read_text(encoding="utf-8"))["per_seed"]}[seed]
        for vname, flags in variants.items():
            if not a.all_variants and vname == "index" and (kind, seed) != ("custom", 1):
                continue
            trace = OUT / "regress" / f"{kind}_s{seed}_{vname}.npz"
            env = UnityVecEnv(EXE, num_agents=20, port=port, log_dir=LOGS, extra_args=flags)
            port += 1
            try:
                build_id = env.hello["build_id"]
                track = (env.hello["track_id"], env.hello["track_index"], env.hello["env_config_hash"])
                res = evaluate(env, spec, seed, 1000, trace)
            finally:
                env.close()
            ps = res["per_seed"]
            (OUT / "regress" / f"{kind}_s{seed}_{vname}.json").write_text(json.dumps(ps, indent=2) + "\n", encoding="utf-8")
            keys = sorted((set(ref) | set(ps)) - SKIP_KEYS)
            metric_diff = [k for k in keys if not same(ref.get(k), ps.get(k))]
            trace_diff = compare_traces(REPO / "benchmarks" / "eval" / "traces" / ref["trace"], trace)
            good = not metric_diff and not trace_diff and track == ("Track_A", 0, "90240ee2b1a58b5b")
            ok &= good
            rows.append({"policy": kind, "seed": seed, "variant": vname, "flags": flags, "ok": good,
                         "flying_lap_median_s": ps["flying_lap_median_s"], "completion_rate": ps["completion_rate"],
                         "flying_laps_identical": same(ref["flying_laps_s"], ps["flying_laps_s"]),
                         "compared_metrics": len(keys), "metric_diff": metric_diff, "trace_diff": trace_diff,
                         "build_id": build_id, "track": list(track)})
            print(f"[regress] {kind:8s} s{seed} {vname:5s} T={ps['flying_lap_median_s']:.2f} "
                  f"(ref {ref['flying_lap_median_s']:.2f}) metrics {'=' if not metric_diff else metric_diff} "
                  f"trace {'=' if not trace_diff else trace_diff} {'OK' if good else 'FAIL'}")
    return {"check": "track_a_regression", "ok": ok, "reference": ["benchmarks/custom_ppo.json", "benchmarks/mlagents_bridge.json",
                                                                   "benchmarks/eval/traces/*.npz"], "runs": rows}


# ---------------------------------------------------------------------------------------------------- determinism
def action_sequence(steps: int, n: int, seed: int) -> np.ndarray:
    """Fixed, car-like actions: smoothed random steering, mostly throttle with some braking."""
    rng = np.random.default_rng(seed)
    a = np.zeros((steps, n, 2), np.float32)
    steer = np.zeros(n)
    for t in range(steps):
        steer = np.clip(0.85 * steer + 0.35 * rng.standard_normal(n), -1, 1)
        a[t, :, 0] = steer
        a[t, :, 1] = rng.uniform(-0.3, 1.0, n)
    return a


def state_digest(obs, rew, term, trunc, info) -> str:
    h = hashlib.sha256()
    for x in (obs, rew, term, trunc, info.get("final_obs", np.zeros(0, np.float32)), *(info[f] for f in INFO_FIELDS)):
        h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def run_process(track: str, a, port: int, mode: int, actions: np.ndarray | None, policy=None) -> dict:
    """actions[t] (fixed sequence) or policy(obs) (deterministic μ: a function of the state, so equal streams stay equal)."""
    expected = FROZEN.get(track, (None, None))[1]
    env = UnityVecEnv(EXE, num_agents=a.agents, port=port, expected_env_hash=expected, log_dir=LOGS,
                      extra_args=["-trackName", track])
    runs = []
    try:
        hello = dict(env.hello)
        for _ in range(2):
            obs, info = env.reset(seed=a.seed, options={"start_mode": mode, "max_laps": 0})
            digests = [state_digest(obs, np.zeros(0), np.zeros(0), np.zeros(0), info)]
            dones = laps = 0
            vmax = 0.0
            fast = np.zeros(20, np.int64)  # agent-decisions above 20 m/s per 1/20 of the track (coverage at speed)
            for t in range(a.steps):
                act = actions[t] if policy is None else np.asarray(policy(obs), np.float32)
                obs, rew, term, trunc, info = env.step(act)
                digests.append(state_digest(obs, rew, term, trunc, info))
                dones += int((term | trunc).sum())
                laps += int(np.sum(info["lap_completed"]))
                vmax = max(vmax, float(np.max(info["speed_mps"])))
                bins = np.clip((info["progress"] * 20).astype(int), 0, 19)
                np.add.at(fast, bins[info["speed_mps"] > 20.0], 1)
            runs.append({"digests": digests, "dones": dones, "laps": laps, "max_speed_mps": vmax,
                         "fast_bins": fast.tolist()})
    finally:
        env.close()
    return {"hello": hello, "runs": runs}


def first_diff(x: list[str], y: list[str]) -> int | None:
    for i, (u, v) in enumerate(zip(x, y)):
        if u != v:
            return i
    return None if len(x) == len(y) else min(len(x), len(y))


def determinism(a) -> dict:
    """Both scenarios spawn over the whole track (TrainRandom, slopes included). 'actions': a fixed action sequence (many
    wall contacts and in-STEP auto-resets). 'policy': the M5 policy (deterministic μ) at racing speed."""
    scenarios = [("actions", a.mode, action_sequence(a.steps, a.agents, a.action_seed), None)]
    if a.policy:
        from racing_rl.train.evaluate import load_policy
        scenarios.append(("policy", a.mode, None, load_policy(a.policy)[0]))
    port, rows, ok = a.port, [], True
    for track in a.tracks:
        for name, mode, actions, policy in scenarios:
            procs = []
            for _ in range(a.processes):
                procs.append(run_process(track, a, port, mode, actions, policy))
                port += 1
            ref = procs[0]["runs"][0]["digests"]
            within = [first_diff(p["runs"][0]["digests"], p["runs"][1]["digests"]) for p in procs]
            across = [first_diff(ref, p["runs"][r]["digests"]) for p in procs[1:] for r in (0, 1)]
            good = all(d is None for d in within + across)
            ok &= good
            final = hashlib.sha256("".join(ref).encode()).hexdigest()[:16]
            rows.append({"track": track, "scenario": name, "start_mode": mode, "env_config_hash": procs[0]["hello"]["env_config_hash"],
                         "ok": good, "processes": a.processes, "resets_per_process": 2, "states_per_run": len(ref),
                         "stream_sha256_16": final, "first_diff_within_process": within, "first_diff_across_processes": across,
                         "dones_per_run": [r["dones"] for p in procs for r in p["runs"]],
                         "laps_per_run": [r["laps"] for p in procs for r in p["runs"]],
                         "max_speed_mps": max(r["max_speed_mps"] for p in procs for r in p["runs"]),
                         "decisions_above_20mps_per_twentieth": procs[0]["runs"][0]["fast_bins"],
                         "build_id": procs[0]["hello"]["build_id"]})
            covered = sum(1 for c in procs[0]["runs"][0]["fast_bins"] if c > 0)
            print(f"[determinism] {track} {name}: {len(ref)} states x {a.agents} agents, stream {final}, within {within}, "
                  f"across {across}, dones/run {rows[-1]['dones_per_run']}, laps/run {rows[-1]['laps_per_run']}, "
                  f"vmax {rows[-1]['max_speed_mps']:.1f} m/s, >20 m/s in {covered}/20 of the track {'OK' if good else 'FAIL'}")
    return {"check": "cross_process_determinism", "ok": ok, "agents": a.agents, "steps": a.steps, "seed": a.seed,
            "action_seed": a.action_seed, "policy": a.policy and Path(a.policy.partition(":")[2]).name, "tracks": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--port", type=int, default=6605)
    s.add_argument("--out", default=str(REPO / "benchmarks" / "eval" / "m6_bridge_smoke.json"))
    r = sub.add_parser("regress")
    r.add_argument("--port", type=int, default=6705)
    r.add_argument("--all-variants", action="store_true")
    r.add_argument("--out", default=str(REPO / "benchmarks" / "eval" / "m6_track_a_regression.json"))
    d = sub.add_parser("determinism")
    d.add_argument("--port", type=int, default=6805)
    d.add_argument("--tracks", nargs="+", default=["Track_D", "Track_A"])
    d.add_argument("--agents", type=int, default=16)
    d.add_argument("--steps", type=int, default=1500)
    d.add_argument("--seed", type=int, default=1000)
    d.add_argument("--mode", type=int, default=0, help="start mode (0 = TrainRandom: spawns over the whole track, 1 = EvalGrid)")
    d.add_argument("--action-seed", type=int, default=0)
    d.add_argument("--policy", default=f"torch:{REPO / 'benchmarks' / 'models' / 'custom_ppo_s1.pt'}",
                   help="policy scenario (deterministic μ); empty string skips it")
    d.add_argument("--processes", type=int, default=2)
    d.add_argument("--out", default=str(REPO / "benchmarks" / "eval" / "m6_track_determinism.json"))
    a = ap.parse_args()
    if not EXE.is_file():
        sys.exit(f"build not found: {EXE}")
    result = {"smoke": smoke, "regress": regress, "determinism": determinism}[a.cmd](a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(("PASS" if result["ok"] else "FAIL") + " -> " + Path(a.out).name)  # the repo path is not ASCII (cp1252 console)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
