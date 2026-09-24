"""
M5 comparison: ML-Agents PPO (M2 ONNX over the bridge) vs custom PyTorch PPO, both evaluated with the same build,
evaluator and seeds (C0.10). Refuses reports whose env_config_hash differ (C0.11).

    python -m racing_rl.train.compare --mlagents ../outputs/benchmarks/mlagents_bridge.json \\
        --custom ../outputs/benchmarks/custom_ppo.json --runs ../outputs/runs/m5/parity_s1 ../outputs/runs/m5/parity_s2 ../outputs/runs/m5/parity_s3 \\
        --notes ../outputs/benchmarks/eval/m5_notes.md --out ../outputs/benchmarks/M5_FINAL.md

Tables: metric | ML-Agents PPO | Custom PPO | Δ%, per-seed results, 1M-decision snapshot, sample efficiency, DoD.
Plots (outputs/benchmarks/plots/): learning curves (reward, training lap time, completion vs decisions; validation flying
lap), flying-lap box plot, racing line (x-z overlay and lateral offset e_lat over s), speed and steering profile
over s, sector-time deltas.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

TOL = 1.01  # DoD: custom T ≤ T_ref × 1.01
MIN_COMPLETION = 0.95
C_ML, C_CU = "#4C72B0", "#DD8452"  # ML-Agents blue, custom orange (same in every plot)


class HashMismatchError(ValueError):
    pass


# --------------------------------------------------------------------------------------------------- helpers
def load(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def med(vals):
    v = [x for x in vals if x is not None]
    return statistics.median(v) if v else None


def fnum(v, nd=2, unit=""):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and v >= 1e4 and float(v).is_integer()):
        return f"{int(v):,}{unit}"
    return f"{v:.{nd}f}{unit}"


def delta(a, b, lower_better=True) -> str:
    if a in (None, 0) or b is None:
        return "—"
    d = (b - a) / abs(a) * 100.0
    good = (d < 0) == lower_better if abs(d) >= 0.005 else None
    mark = "" if good is None else (" ✅" if good else " ⚠️")
    return f"{d:+.2f}%{mark}"


def seed_rows(rep: dict) -> dict[int, dict]:
    return {int(p["seed"]): p for p in rep["per_seed"]}


def aggregate(per_seed: list[dict]) -> dict:
    sec = [p.get("sector_times_s") or [None] * 3 for p in per_seed]
    flying = [p["flying_lap_median_s"] for p in per_seed]
    return {
        "T": med(flying) if all(x is not None for x in flying) else None,
        "completion_med": med([p["completion_rate"] for p in per_seed]),
        "completion_min": min(p["completion_rate"] for p in per_seed),
        "best": min((p["flying_lap_best_s"] for p in per_seed if p["flying_lap_best_s"] is not None), default=None),
        "lap1": med([p["lap1_median_s"] for p in per_seed]),
        "speed": med([p["mean_speed_mps"] for p in per_seed]),
        "steer": med([p["steer_smoothness"] for p in per_seed]),
        "s1": med([s[0] for s in sec]), "s2": med([s[1] for s in sec]), "s3": med([s[2] for s in sec]),
    }


# --------------------------------------------------------------------------------------------------- traces
def flying_lap(tr: dict, agent: int, lap: int = 2) -> dict[str, np.ndarray] | None:
    """Per-decision samples of `agent` during lap `lap` (1-based) of its first episode."""
    done = tr["done"][:, agent]
    first = int(np.argmax(done)) if done.any() else len(done) - 1
    laps = tr["laps"][: first + 1, agent].astype(int)
    idx = np.nonzero(laps == lap - 1)[0]
    if len(idx) < 10:
        return None
    idx = idx[1:]  # the first sample is the line-crossing decision itself (progress may still read ~1)
    out = {k: tr[k][idx, agent].astype(np.float64) for k in ("pos_x", "pos_z", "speed_mps", "progress", "e_lat",
                                                                "steer", "throttle")}
    # progress wraps at the line: drop early samples still reading ~1 and late ones already reading ~0
    prog, i = out["progress"], np.arange(len(out["progress"]))
    half = len(prog) / 2
    keep = (prog < 0.999) & ~((i < half) & (prog > 0.5)) & ~((i >= half) & (prog < 0.5))
    return {k: v[keep] for k, v in out.items()}


def load_traces(dir_: Path, kind: str) -> dict[int, dict]:
    out = {}
    for p in sorted(dir_.glob(f"{kind}_s*.npz")):
        with np.load(p) as z:
            out[int(p.stem.split("_s")[-1])] = {k: z[k] for k in z.files}
    return out


# --------------------------------------------------------------------------------------------------- plots
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9})
    return plt


def run_curve(run: Path, key: str, bin_size: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    """metrics.jsonl → mean of `key` per `bin_size` decisions (ML-Agents summary cadence of the M2 report)."""
    rows = [json.loads(x) for x in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    step = np.array([r["global_step"] for r in rows], np.float64)
    val = np.array([np.nan if r.get(key) is None else r[key] for r in rows], np.float64)
    b = np.ceil(step / bin_size).astype(int)
    xs, ys = [], []
    for k in np.unique(b):
        v = val[b == k]
        v = v[np.isfinite(v)]
        xs.append(k * bin_size)
        ys.append(v.mean() if len(v) else np.nan)
    return np.array(xs), np.array(ys)


def eval_curve(run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = run / "eval.jsonl"
    if not p.exists():
        return np.zeros(0), np.zeros(0), np.zeros(0)
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return (np.array([r["global_step"] for r in rows], float),
            np.array([np.nan if r["flying_lap_median_s"] is None else r["flying_lap_median_s"] for r in rows], float),
            np.array([r["completion_rate"] for r in rows], float))


def plot_learning(ml_train: dict, runs: list[Path], t_ref: float, out: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    panels = [(ax[0, 0], "reward", "ep_return_mean", "Kümülatif ödül (episode)"),
              (ax[0, 1], "lap_time_s", "race_lap_time", "Eğitim tur süresi (s)"),
              (ax[1, 0], "completion_rate", "race_completion_rate", "Eğitim completion (truncation oranı)")]
    for a, mk, ck, title in panels:
        for i, s in enumerate(ml_train.get("per_seed", [])):
            c = s["curve"]
            x = np.array([p["step"] for p in c], float)
            y = np.array([np.nan if p.get(mk) is None else p[mk] for p in c], float)
            a.plot(x / 1e6, y, color=C_ML, alpha=0.9 - 0.25 * i, lw=1.2, label=f"ML-Agents s{i + 1}")
        for i, r in enumerate(runs):
            x, y = run_curve(r, ck)
            a.plot(x / 1e6, y, color=C_CU, alpha=0.9 - 0.25 * i, lw=1.2, ls="-", label=f"Özel PPO {r.name}")
        a.set_title(title)
        a.set_xlabel("Ortam kararı (M)")
    ax[0, 1].axhline(t_ref, color="k", lw=0.8, ls="--", label=f"T_ref {t_ref:.2f} s")
    ax[0, 1].set_ylim(top=min(ax[0, 1].get_ylim()[1], 60))
    a = ax[1, 1]
    for i, r in enumerate(runs):
        x, y, _ = eval_curve(r)
        a.plot(x / 1e6, y, "o-", ms=2.5, color=C_CU, alpha=0.9 - 0.25 * i, lw=1, label=f"{r.name}")
    a.axhline(t_ref, color="k", lw=0.8, ls="--", label=f"T_ref {t_ref:.2f} s")
    a.axhline(t_ref * TOL, color="k", lw=0.6, ls=":", label="T_ref × 1.01")
    a.set_title("Özel PPO doğrulama (tohum 2000..2019) flying lap medyanı")
    a.set_xlabel("Ortam kararı (M)")
    lo = np.nanmin([np.nanmin(eval_curve(r)[1]) if len(eval_curve(r)[1]) else np.nan for r in runs] + [t_ref])
    a.set_ylim(lo - 0.5, lo + 8)
    for a in ax.flat:
        a.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_box(ml: dict, cu: dict, out: Path) -> None:
    plt = _plt()
    seeds = sorted(set(ml) & set(cu))
    fig, a = plt.subplots(figsize=(8, 4))
    data, pos, cols = [], [], []
    for j, s in enumerate(seeds):
        for k, (rep, c) in enumerate(((ml, C_ML), (cu, C_CU))):
            data.append(rep[s].get("flying_laps_s") or [np.nan])
            pos.append(j * 3 + k)
            cols.append(c)
    bp = a.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=True,
                   medianprops={"color": "k", "lw": 1.2})
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    a.set_xticks([j * 3 + 0.5 for j in range(len(seeds))], [f"tohum {s}" for s in seeds])
    a.set_ylabel("Flying lap (s)")
    a.set_title("Flying lap dağılımı (tur 2–3, 20 episode, test tohumları 1000..1019)")
    a.plot([], [], color=C_ML, lw=6, alpha=0.6, label="ML-Agents PPO")
    a.plot([], [], color=C_CU, lw=6, alpha=0.6, label="Özel PPO")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_line(tml: dict, tcu: dict, out_line: Path, out_prof: Path) -> list[dict]:
    """Racing line (x-z and e_lat over s), speed and steering over s: agent 0, flying lap 2, every seed."""
    plt = _plt()
    seeds = sorted(set(tml) & set(tcu))
    fig, ax = plt.subplots(len(seeds), 2, figsize=(12, 3.8 * len(seeds)), squeeze=False,
                           gridspec_kw={"width_ratios": [1, 1.6]})
    fig2, ax2 = plt.subplots(len(seeds), 2, figsize=(12, 3.4 * len(seeds)), squeeze=False)
    stats = []
    for j, s in enumerate(seeds):
        lm, lc = flying_lap(tml[s], 0), flying_lap(tcu[s], 0)
        if lm is None or lc is None:
            continue
        a = ax[j, 0]
        a.plot(lm["pos_x"], lm["pos_z"], color=C_ML, lw=1.4, label="ML-Agents")
        a.plot(lc["pos_x"], lc["pos_z"], color=C_CU, lw=1.0, ls="--", label="Özel PPO")
        a.set_aspect("equal")
        a.set_title(f"Tohum {s}: yarış çizgisi (x–z, ajan 0, tur 2)")
        a.legend(fontsize=7)
        a = ax[j, 1]
        a.plot(lm["progress"] * 100, lm["e_lat"], color=C_ML, lw=1.2, label="ML-Agents")
        a.plot(lc["progress"] * 100, lc["e_lat"], color=C_CU, lw=1.2, label="Özel PPO")
        a.axhline(0, color="k", lw=0.5)
        for b in (100 / 3, 200 / 3):
            a.axvline(b, color="grey", lw=0.6, ls=":")
        a.set_title(f"Tohum {s}: yanal sapma e_lat (m, + sağ) – tur ilerlemesi")
        a.set_xlabel("s / L (%)")
        a.legend(fontsize=7)
        a = ax2[j, 0]
        a.plot(lm["progress"] * 100, lm["speed_mps"] * 3.6, color=C_ML, lw=1.2, label="ML-Agents")
        a.plot(lc["progress"] * 100, lc["speed_mps"] * 3.6, color=C_CU, lw=1.2, label="Özel PPO")
        a.set_title(f"Tohum {s}: hız profili (km/h)")
        a.set_xlabel("s / L (%)")
        a.legend(fontsize=7)
        a = ax2[j, 1]
        a.plot(lm["progress"] * 100, lm["steer"], color=C_ML, lw=1.0, label="ML-Agents")
        a.plot(lc["progress"] * 100, lc["steer"], color=C_CU, lw=1.0, label="Özel PPO")
        a.set_title(f"Tohum {s}: direksiyon aksiyonu a0")
        a.set_xlabel("s / L (%)")
        a.legend(fontsize=7)
        grid = np.linspace(0.01, 0.99, 400)
        interp = lambda l, k: np.interp(grid, l["progress"], l[k])  # noqa: E731
        stats.append({"seed": s,
                      "elat_rms_diff_m": float(np.sqrt(np.mean((interp(lm, "e_lat") - interp(lc, "e_lat")) ** 2))),
                      "vmax_ml_kmh": float(lm["speed_mps"].max() * 3.6), "vmax_cu_kmh": float(lc["speed_mps"].max() * 3.6),
                      "vmin_ml_kmh": float(lm["speed_mps"].min() * 3.6), "vmin_cu_kmh": float(lc["speed_mps"].min() * 3.6),
                      "full_throttle_ml": float((lm["throttle"] > 0.95).mean()),
                      "full_throttle_cu": float((lc["throttle"] > 0.95).mean()),
                      "brake_ml": float((lm["throttle"] < 0).mean()), "brake_cu": float((lc["throttle"] < 0).mean())})
    for f, o in ((fig, out_line), (fig2, out_prof)):
        f.tight_layout()
        f.savefig(o)
        plt.close(f)
    return stats


def plot_sectors(ml: dict, cu: dict, out: Path) -> None:
    plt = _plt()
    seeds = sorted(set(ml) & set(cu))
    fig, a = plt.subplots(figsize=(8, 3.6))
    w = 0.25
    for k in range(3):
        d = [(cu[s]["sector_times_s"][k] or np.nan) - (ml[s]["sector_times_s"][k] or np.nan) for s in seeds]
        a.bar(np.arange(len(seeds)) + (k - 1) * w, d, w, label=f"Sektör {k + 1}", color=["#55A868", "#8172B3",
                                                                                             "#C44E52"][k])
    a.axhline(0, color="k", lw=0.6)
    a.set_xticks(np.arange(len(seeds)), [f"tohum {s}" for s in seeds])
    a.set_ylabel("Δ sektör süresi (s)  (Özel − ML-Agents)")
    a.set_title("Sektör süresi farkları (medyan, flying lap; negatif = özel PPO daha hızlı)")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------- report
def build(ml_rep: dict, cu_rep: dict, runs: list[Path], traces: Path, plots: Path, notes: str | None,
          rel_plots: str) -> str:
    if ml_rep["env_config_hash"] != cu_rep["env_config_hash"]:
        raise HashMismatchError(f"env_config_hash differs: {ml_rep['env_config_hash']} vs {cu_rep['env_config_hash']}")
    if ml_rep["obs_layout_hash"] != cu_rep["obs_layout_hash"]:
        raise HashMismatchError("obs_layout_hash differs")
    ml, cu = seed_rows(ml_rep), seed_rows(cu_rep)
    A, B = aggregate(ml_rep["per_seed"]), aggregate(cu_rep["per_seed"])
    t_ref = A["T"]
    mt, ct = ml_rep.get("training", {}), cu_rep.get("training", {})
    L = []
    w = L.append
    w("# M5 Final Raporu: Özel PyTorch PPO ve ML-Agents PPO Kıyası\n")
    w(f"**env_config_hash:** `{cu_rep['env_config_hash']}` (iki raporda eşit, C0.11) · **obs_layout_hash:** "
      f"`{cu_rep['obs_layout_hash']}` · **Unity:** {cu_rep['unity']} · **build_id:** `{cu_rep['build_id']}` · "
      f"**Değerlendirici:** `{cu_rep['evaluator']}` (ikisi için aynı)\n")
    w(f"C0.10: {cu_rep['eval']['episodes']} episode, EvalGrid, test tohumları {cu_rep['eval']['seed_base']}.."
      f"{cu_rep['eval']['seed_base'] + cu_rep['eval']['episodes'] - 1}, {cu_rep['eval']['laps']} tur, deterministik μ. "
      f"Özel PPO modeli `best.pt`, doğrulama tohumları 2000..2019 üzerinde seçildi (C0.19).\n")

    # ---- DoD
    dod_t = B["T"] is not None and B["T"] <= t_ref * TOL
    dod_c = all(p["completion_rate"] >= MIN_COMPLETION for p in cu_rep["per_seed"])
    w("## DoD\n")
    w("| Ölçüt | Hedef | Sonuç | |")
    w("|---|---|---|---|")
    w(f"| Flying lap medyanlarının tohum medyanı | ≤ T_ref × 1.01 = {t_ref * TOL:.2f} s | {fnum(B['T'])} s "
      f"(T_ref {t_ref:.2f} s) | {'✅' if dod_t else '❌'} |")
    comp = " / ".join(f"{p['completion_rate']:.2f}" for p in cu_rep["per_seed"])
    w(f"| Her tohumda completion_rate | ≥ {MIN_COMPLETION} | {comp} | {'✅' if dod_c else '❌'} |")
    w(f"| env_config_hash eşit | `{ml_rep['env_config_hash']}` | `{cu_rep['env_config_hash']}` | ✅ |")
    w("")
    if notes:
        w(notes.strip() + "\n")

    # ---- head to head (primary: best.pt; the 1M snapshot and the last checkpoint shown next to it)
    others = [("1M", (cu_rep.get("milestone_1000000") or {}).get("per_seed")),
              ("son", (cu_rep.get("final") or {}).get("per_seed"))]
    others = [(n, aggregate(ps)) for n, ps in others if ps]
    w("## Kafa kafaya (tohum medyanları)\n")
    w("Birincil karşılaştırma: **Özel best.pt** (C0.19). 1M anlık görüntüsü ve 10M son checkpoint şeffaflık için "
      "aynı test protokolüyle ayrı sütunlarda.\n")
    hdr = "| Metrik | ML-Agents PPO | **Özel best.pt** | Δ% (best.pt) |" + "".join(
        f" Özel {n} | Δ% ({n}) |" for n, _ in others)
    w(hdr)
    w("|" + "---|" * (4 + 2 * len(others)))
    rows = [("Flying lap medyanı (T, s)", "T", True, 2), ("Flying lap en iyi (s)", "best", True, 2),
            ("1. tur medyanı (s)", "lap1", True, 2), ("completion_rate (medyan)", "completion_med", False, 2),
            ("completion_rate (en düşük)", "completion_min", False, 2), ("Ortalama hız (m/s)", "speed", False, 2),
            ("steer_smoothness (ort. |Δa0|)", "steer", True, 4), ("Sektör 1 (s)", "s1", True, 2),
            ("Sektör 2 (s)", "s2", True, 2), ("Sektör 3 (s)", "s3", True, 2)]
    for name, k, lower, nd in rows:
        w(f"| {name} | {fnum(A[k], nd)} | **{fnum(B[k], nd)}** | {delta(A[k], B[k], lower)} |" + "".join(
            f" {fnum(O[k], nd)} | {delta(A[k], O[k], lower)} |" for _, O in others))
    pad = " |" * (2 * len(others))
    w(f"| İlk 3 tur (karar, eğitim) | {fnum(mt.get('decisions_to_first_3lap'))} | "
      f"{fnum(ct.get('decisions_to_first_3lap'))} | {delta(mt.get('decisions_to_first_3lap'), ct.get('decisions_to_first_3lap'))} |" + pad)
    w(f"| %95 eğitim completion (karar) | {fnum(mt.get('decisions_to_95pct'))} | {fnum(ct.get('decisions_to_95pct'))} | "
      f"{delta(mt.get('decisions_to_95pct'), ct.get('decisions_to_95pct'))} |" + pad)
    w(f"| Toplam eğitim kararı (3 tohum) | {fnum(mt.get('total_env_decisions'))} | {fnum(ct.get('total_env_decisions'))} | |" + pad)
    w(f"| Duvar saati (3 tohum, saat) | {fnum(mt.get('wallclock_h'))} | {fnum(ct.get('wallclock_h'))} | |" + pad)
    w("\nΔ% = (Özel − ML-Agents) / ML-Agents; eğitim satırları koşu düzeyindedir (checkpoint'e bağlı değil). ✅ özel PPO lehine, ⚠️ aleyhine. Örnek verimliliği satırları eğitim "
      "telemetrisinin (M2 ile aynı tanım: `Race/Laps3Rate > 0`, `Race/CompletionRate ≥ 0.95`) tohum medyanıdır.\n")

    # ---- checkpoint matrix (flying lap median, completion) per seed
    mile = {p["seed"]: p for p in (cu_rep.get("milestone_1000000") or {}).get("per_seed", [])}
    fin = {p["seed"]: p for p in (cu_rep.get("final") or {}).get("per_seed", [])}
    cell = lambda p: "—" if p is None else f"{fnum(p['flying_lap_median_s'])} s ({p['completion_rate']:.2f})"  # noqa: E731
    w("### Checkpoint matrisi: flying lap medyanı (completion)\n")
    w("| Tohum | ML-Agents (resmi model) | **Özel best.pt** (adım) | Özel 1M | Özel son checkpoint (adım) |")
    w("|---|---|---|---|---|")
    for s in sorted(set(ml) | set(cu)):
        b, f = cu.get(s), fin.get(s)
        w(f"| {s} | {cell(ml.get(s))} | **{cell(b)}** @{fnum((b or {}).get('checkpoint_step'))} | {cell(mile.get(s))} | "
          f"{cell(f)} @{fnum((f or {}).get('checkpoint_step'))} |")
    w("")

    # ---- per seed
    w("## Tohum bazında (test tohumları 1000..1019)\n")
    w("| Tohum | Politika | Model (adım) | completion | Flying medyan / en iyi (s) | 1. tur (s) | Ort. hız (m/s) | "
      "steer_smoothness | Sektörler (s) | Sonlanma |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    ml_steps = {p["seed"]: p for p in mt.get("per_seed", [])}
    for s in sorted(set(ml) | set(cu)):
        for tag, rep, step in (("ML-Agents", ml.get(s), (ml_steps.get(s) or {}).get("total_env_decisions")),
                               ("Özel", cu.get(s), (cu.get(s) or {}).get("checkpoint_step"))):
            if rep is None:
                continue
            sec = " / ".join(fnum(x) for x in rep["sector_times_s"])
            w(f"| {s} | {tag} | {fnum(step)} | {rep['completion_rate']:.2f} | {fnum(rep['flying_lap_median_s'])} / "
              f"{fnum(rep['flying_lap_best_s'])} | {fnum(rep['lap1_median_s'])} | {fnum(rep['mean_speed_mps'])} | "
              f"{fnum(rep['steer_smoothness'], 4)} | {sec} | {json.dumps(rep['term_reasons'])} |")
    w("")

    # ---- final / milestone checkpoints
    extra_sets = [("Son checkpoint", cu_rep.get("final"))] + [
        (f"{int(k.split('_')[1]):,} karar anlık görüntüsü", v) for k, v in cu_rep.items()
        if k.startswith("milestone_") and v and v.get("per_seed")]
    if any(v for _, v in extra_sets):
        w("## Özel PPO: diğer checkpoint'ler (aynı test protokolü)\n")
        w("| Checkpoint | Tohum | completion | Flying medyan / en iyi (s) | 1. tur (s) | steer_smoothness |")
        w("|---|---|---|---|---|---|")
        for name, v in extra_sets:
            if not v:
                continue
            for p in v["per_seed"]:
                w(f"| {name} | {p['seed']} | {p['completion_rate']:.2f} | {fnum(p['flying_lap_median_s'])} / "
                  f"{fnum(p['flying_lap_best_s'])} | {fnum(p['lap1_median_s'])} | {fnum(p['steer_smoothness'], 4)} |")
            sm = v.get("summary") or {}
            w(f"| **{name} özeti** | | medyan {fnum(sm.get('completion_rate_median'))} | **T = {fnum(sm.get('T_ref_s'))}** | "
              f"| |")
        w("\nML-Agents referansı (1M): s2 41.22 s, s3 41.08 s (1M checkpoint'leri resmi baseline); s1 1M'de 43.31 s "
          "(`eval/mlagents_s1_1M.json`).\n")

    # ---- training
    if ct.get("per_seed"):
        w("## Eğitim koşuları (özel PPO)\n")
        w("| Koşu | Tohum | K × N | Toplam karar | best.pt adımı | İlk 3 tur | %95 completion | Duvar saati (sa) | "
          "Unity restart |")
        w("|---|---|---|---|---|---|---|---|---|")
        for t in ct["per_seed"]:
            w(f"| {t['run_id']} | {t['seed']} | {t['procs']} × {t['agents']} | {fnum(t['total_env_decisions'])} | "
              f"{fnum(t['best_step'])} | {fnum(t['decisions_to_first_3lap'])} | {fnum(t['decisions_to_95pct'])} | "
              f"{fnum(t['wallclock_h'])} | {t['restarts']} |")
        w("")

    # ---- plots
    plots.mkdir(parents=True, exist_ok=True)
    tml, tcu = load_traces(traces, "mlagents"), load_traces(traces, "custom")
    plot_learning(ml_rep.get("training", {}), runs, t_ref, plots / "learning_curves.png")
    plot_box(ml, cu, plots / "lap_time_box.png")
    plot_sectors(ml, cu, plots / "sector_deltas.png")
    line_stats = plot_line(tml, tcu, plots / "racing_line.png", plots / "speed_steer_profile.png") if tml and tcu else []
    if line_stats:
        w("## Sürüş karakteri (ajan 0, tur 2)\n")
        w("| Tohum | e_lat RMS farkı (m) | V_max ML / Özel (km/h) | V_min ML / Özel (km/h) | Tam gaz payı ML / Özel | "
          "Fren payı ML / Özel |")
        w("|---|---|---|---|---|---|")
        for r in line_stats:
            w(f"| {r['seed']} | {r['elat_rms_diff_m']:.2f} | {r['vmax_ml_kmh']:.1f} / {r['vmax_cu_kmh']:.1f} | "
              f"{r['vmin_ml_kmh']:.1f} / {r['vmin_cu_kmh']:.1f} | {r['full_throttle_ml']:.2f} / "
              f"{r['full_throttle_cu']:.2f} | {r['brake_ml']:.2f} / {r['brake_cu']:.2f} |")
        w("")
    w("## Grafikler\n")
    for f, cap in (("learning_curves.png", "Öğrenme eğrileri (x: ortam kararı; ML-Agents 100k özet aralığı, özel PPO "
                                           "100k kutulara ortalanmış)"),
                   ("lap_time_box.png", "Flying lap kutu grafiği"),
                   ("racing_line.png", "Yarış çizgisi bindirmesi ve yanal sapma"),
                   ("speed_steer_profile.png", "Hız ve direksiyon profili"),
                   ("sector_deltas.png", "Sektör süresi farkları")):
        if (plots / f).exists():
            w(f"**{cap}**\n\n![{cap}]({rel_plots}/{f})\n")
    w("## Kaynak dosyalar\n")
    w("- `mlagents_bridge.json`, `custom_ppo.json`: C0.10 şeması (`evaluator: bridge`).")
    w("- `eval/traces/*.npz`: karar başına iz (`racing_rl.train.evaluate --trace`).")
    w("- Yeniden üretme: `python scripts/m5_benchmark.py --runs ...` ardından `python -m racing_rl.train.compare ...`.")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mlagents", required=True)
    ap.add_argument("--custom", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--traces")
    ap.add_argument("--plots")
    ap.add_argument("--notes")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    traces = Path(a.traces) if a.traces else Path(a.custom).parent / "eval" / "traces"
    plots = Path(a.plots) if a.plots else out.parent / "plots"
    notes = Path(a.notes).read_text(encoding="utf-8") if a.notes else None
    md = build(load(a.mlagents), load(a.custom), [Path(r) for r in a.runs], traces, plots, notes,
               Path(plots).resolve().relative_to(out.parent.resolve()).as_posix())
    out.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
