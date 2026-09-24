"""
Repository paths shared with Unity. paths.json at the repo root is the single source for every cross-language path
(player builds, benchmark outputs, run logs, test results); Racing.Editor.RepoPaths reads the same file. The root is
found by walking up from this file, then from the working directory; RACEAGENT_ROOT overrides both.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PATHS_FILE = "paths.json"


def _find_root() -> Path:
    env = os.environ.get("RACEAGENT_ROOT")
    if env:
        return Path(env).resolve()
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for d in (start, *start.parents):
            if (d / PATHS_FILE).is_file():
                return d
    raise FileNotFoundError(f"{PATHS_FILE} not found above {Path(__file__).resolve()} or {Path.cwd()}; set RACEAGENT_ROOT")


REPO_ROOT = _find_root()
_CFG: dict[str, str] = json.loads((REPO_ROOT / PATHS_FILE).read_text(encoding="utf-8"))


def repo_path(key: str) -> Path:
    """Absolute path for a paths.json key."""
    return REPO_ROOT / _CFG[key]


def rel(p: str | os.PathLike) -> str:
    """Repo-relative POSIX form of an absolute path under the repo (for reports and policy specs)."""
    return Path(p).relative_to(REPO_ROOT).as_posix()


UNITY_PROJECT = repo_path("unity_project")
BRIDGE_EXE = repo_path("bridge_exe")
MLA_EXE = repo_path("mla_exe")
BENCHMARKS = repo_path("benchmarks")
MODELS = BENCHMARKS / "models"
EVAL = BENCHMARKS / "eval"
TRACES = EVAL / "traces"
PLOTS = BENCHMARKS / "plots"
RUNS = repo_path("runs")
TEST_RESULTS = repo_path("test_results")
TRACK_CATALOG = repo_path("track_catalog")
