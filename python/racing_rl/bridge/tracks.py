"""
M6 track catalog on the Python side (contracts C0.20). track_catalog.json is exported from Unity (menu
Racing/Tracks/Export Track Catalog (M6)); the EditMode TrackCatalogExportTests keep it in sync with the assets.

Names resolve like Racing.Core.TrackCatalog.TryResolve: a catalog id ("Track_B"), an asset name ("TrackDefinition_B")
or "proc:<seed>" (non-negative decimal int64, ProceduralTrackGenerator; index -1, not in the catalog). Unknown names
fail here, before a Unity process is started. The expected env_config_hash of a track comes from the catalog; for
"proc:<seed>" only the frozen reference seeds (procedural_refs) have one, other seeds are recorded from HELLO.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

CATALOG_PATH = Path(__file__).with_name("track_catalog.json")
CATALOG_SCHEMA = "race-track-catalog/v1"
BENCHMARK_ID = "Track_A"
PROC_PREFIX = "proc:"
MULTI_PREFIX = "multi:"
_INT64_MAX = 2**63 - 1


class UnknownTrackError(ValueError):
    """The name is neither a catalog track nor proc:<seed> (Python-side mirror of the bridge's UNKNOWN_TRACK)."""


@dataclass(frozen=True)
class TrackEntry:
    """One exported track: the HELLO track fields plus the frozen hashes (index -1 for procedural references)."""

    index: int
    id: str
    profile: str
    width: float
    length_m: float
    checkpoints: int
    half_width: float
    elevation: bool
    env_config_hash: str
    track_hash: str
    asset: str | None = None


@dataclass(frozen=True)
class TrackSpec:
    """A resolved track name: the id passed as -trackName / expected_track_id and what HELLO must report."""

    id: str
    index: int
    entry: TrackEntry | None  # catalog track or frozen procedural reference; None for any other proc seed

    @property
    def expected_env_hash(self) -> str | None:
        return None if self.entry is None else self.entry.env_config_hash

    @property
    def procedural(self) -> bool:
        return self.index < 0


@dataclass(frozen=True)
class TrackCatalog:
    tracks: tuple[TrackEntry, ...]
    procedural_refs: tuple[TrackEntry, ...]
    obs_layout_hash: str
    generator_version: int

    @property
    def ids(self) -> list[str]:
        return [t.id for t in self.tracks]

    def entry(self, track_id: str) -> TrackEntry | None:
        """Catalog track or procedural reference with this id."""
        for t in (*self.tracks, *self.procedural_refs):
            if t.id == track_id:
                return t
        return None

    def env_hashes(self) -> dict[str, str]:
        """env_config_hash -> track id over the catalog and the procedural references."""
        return {t.env_config_hash: t.id for t in (*self.tracks, *self.procedural_refs)}


@functools.lru_cache(maxsize=4)
def load_catalog(path: str | Path = CATALOG_PATH) -> TrackCatalog:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != CATALOG_SCHEMA:
        raise ValueError(f"{Path(path).name}: schema {data.get('schema')!r} != {CATALOG_SCHEMA!r}")

    def entries(key: str) -> tuple[TrackEntry, ...]:
        return tuple(TrackEntry(**e) for e in data[key])

    cat = TrackCatalog(entries("tracks"), entries("procedural_refs"), data["obs_layout_hash"],
                       int(data["procedural_generator_version"]))
    if [t.index for t in cat.tracks] != list(range(len(cat.tracks))) or cat.tracks[0].id != BENCHMARK_ID:
        raise ValueError(f"{Path(path).name}: catalog indices must be 0..n-1 with {BENCHMARK_ID} at 0")
    return cat


def parse_proc_seed(name: str) -> int | None:
    """Seed of "proc:<seed>" (ProceduralTrackGenerator.TryParseName: 1-19 ASCII digits, no sign, fits int64), else None."""
    if not isinstance(name, str) or not name.startswith(PROC_PREFIX):
        return None
    digits = name[len(PROC_PREFIX):]
    if not 1 <= len(digits) <= 19 or any(not "0" <= ch <= "9" for ch in digits):
        return None
    seed = int(digits)
    return seed if seed <= _INT64_MAX else None


def resolve_track(name: str) -> TrackSpec:
    """Catalog id, asset name or proc:<seed> -> TrackSpec (proc ids are canonical: "proc:007" -> "proc:7")."""
    cat = load_catalog()
    seed = parse_proc_seed(name)
    if seed is not None:
        tid = f"{PROC_PREFIX}{seed}"
        return TrackSpec(tid, -1, cat.entry(tid))
    for t in cat.tracks:
        if name in (t.id, t.asset):
            return TrackSpec(t.id, t.index, t)
    raise UnknownTrackError(f"unknown track {name!r}: expected one of {cat.ids} or proc:<seed>")


def composite_hash(tracks: Mapping[str, str]) -> str:
    """
    env_config_hash of a run over several tracks: "multi:" + the first 16 hex digits of the SHA-256 of the sorted,
    unique "id=hash" lines. A single track is its own hash, so single-track checkpoints keep a plain env_config_hash.
    """
    items = sorted(set(tracks.items()))
    if not items:
        raise ValueError("composite_hash needs at least one track")
    if len(items) == 1:
        return items[0][1]
    text = "".join(f"{tid}={h}\n" for tid, h in items)
    return MULTI_PREFIX + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def training_tracks(env_config_hash: str, recorded: Mapping[str, str] | None) -> dict[str, str]:
    """
    Tracks a checkpoint was trained on, validated against the catalog (the C0.11 guard for M6; raises ValueError).
    With extra.tracks: every catalog / frozen-procedural id must carry its catalog hash (other proc seeds are taken as
    recorded) and composite_hash(extra.tracks) must be the checkpoint's env_config_hash. Without (M4/M5 checkpoints):
    env_config_hash must be the hash of a catalog track or a frozen procedural reference. Any Sim/Vehicle/Reward/obs
    change alters every catalog hash, so the Env Freeze check survives.
    """
    if recorded:
        out = {}
        for tid, h in recorded.items():
            try:
                spec = resolve_track(tid)
            except UnknownTrackError as e:
                raise ValueError(f"trained on {e}") from e
            if spec.expected_env_hash is not None and h != spec.expected_env_hash:
                raise ValueError(f"extra.tracks[{tid}] = {h}, catalog says {spec.expected_env_hash}")
            out[spec.id] = h
        if composite_hash(out) != env_config_hash:
            raise ValueError(f"env_config_hash {env_config_hash} != composite of extra.tracks {composite_hash(out)}")
        return out
    known = load_catalog().env_hashes()
    if env_config_hash not in known:
        raise ValueError(f"env_config_hash {env_config_hash} is not a catalog track hash (C0.20) and the "
                         "checkpoint has no extra.tracks")
    return {known[env_config_hash]: env_config_hash}
