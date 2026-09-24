"""
M5 DoD "reproducible runs": two fresh train_ppo runs (fresh Unity processes) with the same config and seed must give
bit-identical first-update metrics (timing keys excluded; floats compared with float.hex).

    python scripts/m5_repro_check.py --config configs/ppo_parity.yaml --seed 1 --updates 2
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racing_rl import paths  # noqa: E402
from racing_rl.train import train_ppo  # noqa: E402

TIMING = {"time_collect_s", "time_update_s", "sps", "collect_sps", "wallclock_s", "step_latency_p50_ms",
          "step_latency_p99_ms", "time_eval_s"}


def canon(v):
    return float(v).hex() if isinstance(v, float) else v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "ppo_parity.yaml"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--updates", type=int, default=2)
    ap.add_argument("--out", default=str(paths.EVAL / "m5_repro.json"))
    a = ap.parse_args()
    root = paths.RUNS / "m5" / "_repro"
    shutil.rmtree(root, ignore_errors=True)
    hist = []
    for i in range(2):
        rd = root / f"r{i}"
        rc = train_ppo.main(["--config", a.config, "--seed", str(a.seed), "--run-dir", str(rd), "--eval-every", "0",
                             "--max-updates", str(a.updates), "--no-tensorboard", "--milestones"])
        assert rc == 0
        hist.append([json.loads(x) for x in (rd / "metrics.jsonl").read_text(encoding="utf-8").splitlines()])
    result = {"config": Path(a.config).name, "seed": a.seed, "updates": a.updates, "per_update": []}
    ok = True
    for u in range(a.updates):
        x, y = hist[0][u], hist[1][u]
        keys = sorted(k for k in x if k not in TIMING)
        diff = [k for k in keys if canon(x[k]) != canon(y[k])]
        ok &= not diff
        result["per_update"].append({"update": u + 1, "compared_keys": len(keys), "different": diff,
                                     "policy_loss": x["policy_loss"], "value_loss": x["value_loss"],
                                     "approx_kl": x["approx_kl"], "sps": [x["sps"], y["sps"]]})
    result["identical"] = ok
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
