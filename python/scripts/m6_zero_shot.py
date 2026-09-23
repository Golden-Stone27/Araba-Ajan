"""
M6 zero-shot generalisation benchmark (docs/milestones/M6_multitrack.md step 5, contracts C0.20): the six M5 models,
all trained on Track_A only (custom PPO best.pt s1-s3, ML-Agents ONNX s1-s3), evaluated with the C0.10 protocol on
every catalog track and four procedural tracks.

    python scripts/m6_zero_shot.py run [--tracks Track_B Track_D]
    python scripts/m6_zero_shot.py report

run     C0.10 over the bridge (EvalGrid, seeds 1000..1019, 3 laps, deterministic μ, first episode per agent) through
        racing_rl.train.evaluate with UnityVecEnv(track=T): one 20-agent process per track, reused for the six models
        (every evaluation starts with RESET, which rebuilds the cars, C0.17; in-process RESET is bit-reproducible on
        Track_D too, C0.20), as in the M5 benchmark. Track_A goes through the same path and must reproduce the M5
        per-seed metrics exactly (benchmarks/custom_ppo.json, mlagents_bridge.json). Per-decision traces go to
        runs/m6/zero_shot/traces (gitignored); where and why every first episode ended (reason, laps, s in m) is
        extracted from them. Results: benchmarks/eval/m6_zero_shot.json (reruns of a subset of tracks are merged).
report  benchmarks/M6_TRACKS.md: tracks, head-to-head per track (completion, flying lap T and T / PurePursuit,
        lap 1, speed, distance driven), termination reasons, crash locations (Track_D by elevation section), the
        per-seed matrix; the summary text comes from benchmarks/eval/m6_notes.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from racing_rl.bridge.evaluate import EPISODES, LAPS, SEED_BASE, report as bench_report  # noqa: E402
from racing_rl.bridge.protocol import TERM_REASON_KEYS, TermReason  # noqa: E402
from racing_rl.bridge.tracks import load_catalog, resolve_track  # noqa: E402
from racing_rl.bridge.vec_env import UnityVecEnv  # noqa: E402
from racing_rl.train.evaluate import DEFAULT_EXE, evaluate  # noqa: E402

OUT = REPO / "runs" / "m6" / "zero_shot"
LOGS = OUT / "unity_logs"
RESULT = REPO / "benchmarks" / "eval" / "m6_zero_shot.json"
REPORT = REPO / "benchmarks" / "M6_TRACKS.md"
NOTES = REPO / "benchmarks" / "eval" / "m6_notes.md"
SCHEMA = "race-m6-zero-shot/v1"

TRACKS = ["Track_A", "Track_B", "Track_C", "Track_D", "proc:0", "proc:1", "proc:7", "proc:1000"]
FAMILIES = {"custom": ("custom-ppo", "custom_ppo.json"), "mlagents": ("mlagents-ppo", "mlagents_bridge.json")}
MODELS = [(kind, s, spec) for kind, fmt in (("custom", "torch:{}/custom_ppo_s{}.pt"),
                                            ("mlagents", "onnx:{}/mlagents_baseline_s{}.onnx"))
          for s in (1, 2, 3) for spec in [fmt.format(REPO / "benchmarks" / "models", s)]]
SKIP_KEYS = {"wallclock_s", "trace", "policy_spec", "checkpoint_step", "model"}  # run metadata, not results

# PurePursuit laps from the grid, seed 1000 (PlayMode *PurePursuit_Completes3Laps_WithoutWallContact, C0.20)
PUREPURSUIT = {
    "Track_A": (54.58, 51.78, 51.78), "Track_B": (67.08, 65.02, 65.00), "Track_C": (44.14, 40.28, 40.26),
    "Track_D": (53.96, 51.86, 51.86), "proc:0": (57.72, 55.88, 55.90), "proc:1": (59.04, 56.22, 56.24),
    "proc:7": (47.62, 45.16, 45.16), "proc:1000": (47.00, 44.44, 44.44),
}
# Track_D elevation profile along s (C0.20): flat to 150 m, +10 m at 480 m, plateau to 600 m, back to 0 at 960 m
HILL_D = [(0.0, 150.0, "düz (başlangıç)"), (150.0, 480.0, "yokuş çıkışı"), (480.0, 600.0, "plato"),
          (600.0, 960.0, "iniş"), (960.0, 1e9, "düz (bitiş)")]


# ---------------------------------------------------------------------------------------------------- run
def episode_ends(trace: Path, length_m: float) -> dict:
    """First episode of every agent: termination reason, completed laps and position s (m) at the end."""
    z = np.load(trace)
    done = z["done"]
    out = {"reason": [], "laps": [], "s_m": []}
    for i in range(done.shape[1]):
        idx = np.flatnonzero(done[:, i])
        t = int(idx[0]) if len(idx) else done.shape[0] - 1
        reason = TermReason(int(z["term_reason"][t, i])) if len(idx) else TermReason.NONE
        out["reason"].append(TERM_REASON_KEYS[reason])
        out["laps"].append(int(z["laps"][t, i]))
        out["s_m"].append(round(float(z["progress"][t, i]) * length_m, 1))
    return out


def same(a, b) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def run(a) -> int:
    data = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.is_file() else {"schema": SCHEMA, "tracks": {}}
    refs = {kind: {p["seed"]: p for p in json.loads((REPO / "benchmarks" / f).read_text(encoding="utf-8"))["per_seed"]}
            for kind, (_, f) in FAMILIES.items()}
    ok = True
    for ti, name in enumerate(a.tracks):
        track = resolve_track(name).id
        env = UnityVecEnv(a.exe, num_agents=EPISODES, port=a.port + ti, log_dir=LOGS, track=track)
        t0 = time.perf_counter()
        try:
            info = dict(env.track_info)
            per_family: dict[str, list] = {k: [] for k in FAMILIES}
            ends, checks, train_tracks = {}, {}, {}
            for kind, seed, spec in MODELS:
                trace = OUT / "traces" / f"{track.replace(':', '')}_{kind}_s{seed}.npz"
                res = evaluate(env, spec, seed, SEED_BASE, trace)
                ps = res["per_seed"]
                ps["policy_spec"] = spec.split(":", 1)[0] + ":benchmarks/models/" + Path(spec.split(":", 1)[1]).name
                per_family[kind].append(ps)
                train_tracks[f"{kind}_s{seed}"] = res["train_tracks"]
                ends[f"{kind}_s{seed}"] = episode_ends(trace, info["track_length_m"])
                line = (f"[{track}] {kind:8s} s{seed} completion {ps['completion_rate']:.2f} "
                        f"T {ps['flying_lap_median_s'] or float('nan'):.2f} {ps['term_reasons']}")
                if track == "Track_A":  # the M5 benchmark through the new track API: must be identical
                    ref = refs[kind][seed]
                    diff = [k for k in sorted((set(ref) | set(ps)) - SKIP_KEYS) if not same(ref.get(k), ps.get(k))]
                    checks[f"{kind}_s{seed}"] = {"m5_identical": not diff, "diff": diff}
                    ok &= not diff
                    line += " M5 " + ("identical" if not diff else f"DIFF {diff}")
                print(line, flush=True)
            reports = {}
            for kind, (pname, _) in FAMILIES.items():
                rep = bench_report(env, per_family[kind], pname)
                rep["eval"] = {"episodes": EPISODES, "seed_base": SEED_BASE, "laps": LAPS, "start": "grid",
                               "deterministic": True}
                rep["summary"] = family_summary(per_family[kind])
                reports[kind] = rep
            data["tracks"][track] = {"track": info, "purepursuit_laps_s": list(PUREPURSUIT.get(track, ())),
                                     "reports": reports, "episode_ends": ends, "train_tracks": train_tracks,
                                     "wallclock_s": round(time.perf_counter() - t0, 1),
                                     **({"m5_check": checks} if checks else {})}
            data.update(build_id=env.hello["build_id"], unity=env.hello["unity"],
                        obs_layout_hash=env.hello["obs_layout_hash"])
        finally:
            env.close()
        RESULT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")  # after every track: partial results
    data["protocol"] = {"episodes": EPISODES, "seed_base": SEED_BASE, "laps": LAPS, "start": "grid",
                        "deterministic": True, "first_episode_only": True}
    data["models"] = [spec.split(":", 1)[0] + ":benchmarks/models/" + Path(spec.split(":", 1)[1]).name
                      for _, _, spec in MODELS]
    data["tracks"] = {t: data["tracks"][t] for t in TRACKS if t in data["tracks"]} | {
        t: v for t, v in data["tracks"].items() if t not in TRACKS}
    RESULT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(("PASS" if ok else "FAIL (Track_A differs from M5)") + " -> " + RESULT.name)
    return 0 if ok else 1


def family_summary(per_seed: list[dict]) -> dict:
    """C0.10 T_ref definition: median over seeds of the per-seed flying-lap medians (seeds without a finish skipped)."""
    fl = [p["flying_lap_median_s"] for p in per_seed if p["flying_lap_median_s"] is not None]
    comp = [p["completion_rate"] for p in per_seed]
    return {"T_s": statistics.median(fl) if fl else None, "T_seeds": len(fl),
            "completion_rate_median": statistics.median(comp), "completion_rate_min": min(comp)}


# ---------------------------------------------------------------------------------------------------- report
def fmt(v, spec=".2f", none="—"):
    return none if v is None else format(v, spec)


def pp_flying(track: str) -> float | None:
    laps = PUREPURSUIT.get(track)
    return statistics.median(laps[1:]) if laps else None


def pooled_completion(per_seed: list[dict]) -> float:
    return sum(p["completion_rate"] * p["episodes"] for p in per_seed) / sum(p["episodes"] for p in per_seed)


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def family_rows(t: dict, kind: str) -> dict:
    rep = t["reports"][kind]
    ps = rep["per_seed"]
    ends = [t["episode_ends"][f"{kind}_s{p['seed']}"] for p in ps]
    length = t["track"]["track_length_m"]
    driven = [lap + s / length for e in ends for lap, s, r in zip(e["laps"], e["s_m"], e["reason"])
              if r != "finished"]
    reasons = Counter()
    for p in ps:
        reasons.update(p["term_reasons"])
    crashes = [s for e in ends for s, r in zip(e["s_m"], e["reason"]) if r != "finished"]
    return {"summary": rep["summary"], "per_seed": ps, "reasons": reasons, "crashes": crashes,
            "lap1": med([p["lap1_median_s"] for p in ps]), "speed": med([p["mean_speed_mps"] for p in ps]),
            "laps_before_end": med(driven), "episodes": sum(p["episodes"] for p in ps)}


def crash_clusters(crashes: list[float], gap_m: float = 15.0) -> str:
    """Crash positions grouped where consecutive sorted positions lie within gap_m: "≈657 m ×40, ≈182 m ×20"."""
    if not crashes:
        return "—"
    groups: list[list[float]] = []
    for s in sorted(crashes):
        if groups and s - groups[-1][-1] <= gap_m:
            groups[-1].append(s)
        else:
            groups.append([s])
    groups.sort(key=lambda g: (-len(g), g[0]))
    return ", ".join(f"≈{statistics.median(g):.0f} m ×{len(g)}" for g in groups)


def build_report(data: dict, notes: str) -> str:
    cat = load_catalog()
    tracks = data["tracks"]
    L: list[str] = []
    w = L.append
    w("# M6 Sıfır Atış Raporu: M5 Modelleri Yeni Pistlerde")
    w("")
    w("> **Soru:** Yalnız Track_A'da eğitilmiş M5 modelleri, hiç görmedikleri pistlerde ne kadar sürebiliyor?")
    w(f"> **Protokol:** C0.10 (EvalGrid, tohumlar {SEED_BASE}..{SEED_BASE + EPISODES - 1}, {LAPS} tur, deterministik μ, "
      "ajan başına ilk episode), köprü değerlendiricisi (`racing_rl.train.evaluate`, `UnityVecEnv(track=...)`).")
    w(f"> **Modeller:** özel PPO `custom_ppo_s{{1,2,3}}.pt` (M5 `best.pt`) ve ML-Agents `mlagents_baseline_s{{1,2,3}}.onnx` "
      "(M2). Hepsi yalnız Track_A'da eğitildi. **Build:** `Builds/RaceEnv` "
      f"(`build_id` `{data.get('build_id', '?')}`, Unity {data.get('unity', '?')}). "
      "Üretim: `python scripts/m6_zero_shot.py run` ve `report`; ham sonuçlar `benchmarks/eval/m6_zero_shot.json`.")
    w("")
    if notes.strip():
        w(notes.strip())
        w("")

    w("## Pistler")
    w("")
    w("| Pist | Profil | L (m) | W (m) | Kapı | Kot | PurePursuit turları (s) | PP flying (s) | `env_config_hash` |")
    w("|---|---|---|---|---|---|---|---|---|")
    for tid, t in tracks.items():
        i = t["track"]
        e = cat.entry(tid)
        pp = t["purepursuit_laps_s"]
        w(f"| {tid} | {i['profile']} | {i['track_length_m']:.1f} | {2 * i['track_half_width']:g} | "
          f"{i['track_checkpoints']} | {'+10 m' if e and e.elevation else '—'} | "
          f"{' / '.join(f'{x:.2f}' for x in pp) if pp else '—'} | {fmt(pp_flying(tid))} | `{i['env_config_hash']}` |")
    w("")

    w("## Kafa kafaya (3 eğitim tohumu)")
    w("")
    w("Completion: 60 episode üzerinden (3 tohum × 20); parantezde en az bir episode'u bitiren tohum sayısı. "
      "T: bu tohumların flying lap medyanlarının medyanı (C0.10 T_ref tanımı). T/PP: aynı pistte PurePursuit flying "
      "lap'ine oran (Track_A satırı referanstır). Mesafe: bitiremeyen episode'ların sonlanmadan önce sürdüğü tur (medyan).")
    w("")
    w("| Pist | Completion özel | Completion ML | T özel (s) | T ML (s) | T/PP özel | T/PP ML "
      "| 1. tur özel / ML (s) | Ort. hız özel / ML (m/s) | Mesafe özel / ML (tur) |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for tid, t in tracks.items():
        c, m = family_rows(t, "custom"), family_rows(t, "mlagents")
        cs, ms = c["summary"], m["summary"]
        pp = pp_flying(tid)
        ratio = lambda s: fmt(s["T_s"] / pp if s["T_s"] and pp else None, ".3f")  # noqa: E731
        tcol = lambda s: fmt(s["T_s"])  # noqa: E731
        comp = lambda r: f"{pooled_completion(r['per_seed']):.2f} ({r['summary']['T_seeds']}/3)"  # noqa: E731
        w(f"| {tid} | {comp(c)} | {comp(m)} | {tcol(cs)} | {tcol(ms)} | "
          f"{ratio(cs)} | {ratio(ms)} | {fmt(c['lap1'])} / {fmt(m['lap1'])} | {fmt(c['speed'])} / {fmt(m['speed'])} | "
          f"{fmt(c['laps_before_end'])} / {fmt(m['laps_before_end'])} |")
    w("")

    w("## Bitiş nedenleri (60 episode = 3 tohum × 20)")
    w("")
    w("| Pist | Özel PPO | ML-Agents |")
    w("|---|---|---|")
    for tid, t in tracks.items():
        cols = []
        for kind in FAMILIES:
            r = family_rows(t, kind)["reasons"]
            cols.append(", ".join(f"{k} {v}" for k, v in sorted(r.items(), key=lambda kv: (-kv[1], kv[0]))))
        w(f"| {tid} | {cols[0]} | {cols[1]} |")
    w("")

    w("## Çarpışma konumu")
    w("")
    w("Bitiremeyen ilk episode'ların sonlandığı konum (s, start çizgisinden metre). Konumlar 15 m içinde art arda "
      "geliyorsa aynı kümeye girer; küme başına medyan s ve episode sayısı.")
    w("")
    w("| Pist | Bitiremeyen özel / ML | Kümeler özel | Kümeler ML |")
    w("|---|---|---|---|")
    for tid, t in tracks.items():
        c, m = family_rows(t, "custom"), family_rows(t, "mlagents")
        w(f"| {tid} | {len(c['crashes'])} / {len(m['crashes'])} | {crash_clusters(c['crashes'])} | "
          f"{crash_clusters(m['crashes'])} |")
    w("")
    if "Track_D" in tracks:
        w("**Track_D kot bölgeleri** (C0.20 profili; bölge sınırları yaklaşıktır):")
        w("")
        w("| Bölge | s (m) | Özel PPO | ML-Agents |")
        w("|---|---|---|---|")
        t = tracks["Track_D"]
        crashes = {k: family_rows(t, k)["crashes"] for k in FAMILIES}
        for lo, hi, name in HILL_D:
            n = [sum(lo <= s < hi for s in crashes[k]) for k in FAMILIES]
            span = f"{lo:.0f}–{hi:.0f}" if hi < 1e8 else f"{lo:.0f}–L"
            w(f"| {name} | {span} | {n[0]} | {n[1]} |")
        w("")

    w("## Tohum bazında matris")
    w("")
    w("Hücre: completion · flying lap medyanı (s).")
    w("")
    w("| Pist | Özel s1 | Özel s2 | Özel s3 | ML s1 | ML s2 | ML s3 |")
    w("|---|---|---|---|---|---|---|")
    for tid, t in tracks.items():
        cells = [f"{p['completion_rate']:.2f} · {fmt(p['flying_lap_median_s'])}"
                 for kind in FAMILIES for p in t["reports"][kind]["per_seed"]]
        w(f"| {tid} | " + " | ".join(cells) + " |")
    w("")

    if "Track_A" in tracks and "m5_check" in tracks["Track_A"]:
        chk = tracks["Track_A"]["m5_check"]
        good = all(v["m5_identical"] for v in chk.values())
        w("## Track_A kontrolü")
        w("")
        w(f"Track_A, yeni pist API'si (`UnityVecEnv(track=\"Track_A\")`: `-trackName`, CONFIG `expected_track_id`) ve aynı "
          "süreçte 6 modelle koşuldu. Tüm per-seed metrikleri M5 raporlarıyla "
          f"(`benchmarks/custom_ppo.json`, `benchmarks/mlagents_bridge.json`) {'**bit düzeyinde aynı**' if good else '**FARKLI**'} "
          f"({sum(v['m5_identical'] for v in chk.values())}/{len(chk)}).")
        w("")
    return "\n".join(L) + "\n"


def make_report(a) -> int:
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    notes = Path(a.notes).read_text(encoding="utf-8") if a.notes and Path(a.notes).is_file() else ""
    REPORT.write_text(build_report(data, notes), encoding="utf-8")
    print("-> " + REPORT.name)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tracks", nargs="+", default=TRACKS)
    r.add_argument("--exe", default=str(DEFAULT_EXE))
    r.add_argument("--port", type=int, default=6805)
    p = sub.add_parser("report")
    p.add_argument("--notes", default=str(NOTES))
    a = ap.parse_args()
    return run(a) if a.cmd == "run" else make_report(a)


if __name__ == "__main__":
    sys.exit(main())
