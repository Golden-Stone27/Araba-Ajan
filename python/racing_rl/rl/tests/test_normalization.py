"""DoD 6: RunningMeanStd chunked / parallel updates match numpy on the concatenated data (≤ 1e-6 relative)."""

import numpy as np
import pytest
import torch

from racing_rl.rl import ActorCritic, RunningMeanStd


def _data(rng):
    scale = rng.uniform(0.01, 50.0, 26)
    loc = rng.uniform(-100.0, 100.0, 26)
    return rng.normal(size=(20_000, 26)) * scale + loc


def _rel(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


def test_chunked_sequential_updates(rng):
    x = _data(rng)
    cuts = np.sort(rng.choice(np.arange(1, len(x)), 40, replace=False))
    rms = RunningMeanStd(26)
    for chunk in np.split(x, cuts):
        rms.update(chunk)
    assert rms.count == len(x)
    assert _rel(rms.mean, x.mean(0)) <= 1e-6
    assert _rel(rms.var, x.var(0)) <= 1e-6


def test_parallel_merge(rng):
    x = _data(rng)
    parts = np.array_split(x, 7)
    workers = []
    for p in parts:
        w = RunningMeanStd(26)
        for c in np.array_split(p, 5):
            w.update(c)
        workers.append(w)
    total = RunningMeanStd(26)
    for w in workers:
        total.merge(w)
    assert _rel(total.mean, x.mean(0)) <= 1e-6
    assert _rel(total.var, x.var(0)) <= 1e-6


def test_normalize_and_state_dict(rng):
    x = _data(rng)
    rms = RunningMeanStd(26)
    rms.update(x)
    z = rms.normalize(x[:5], clip=10.0)
    assert np.allclose(z, np.clip((x[:5] - x.mean(0)) / np.sqrt(x.var(0) + 1e-8), -10, 10), rtol=1e-6)
    assert np.abs(rms.normalize(np.full((1, 26), 1e9))).max() == 10.0
    other = RunningMeanStd(26)
    other.load_state_dict(rms.state_dict())
    assert np.array_equal(other.mean, rms.mean) and np.array_equal(other.var, rms.var) and other.count == rms.count


def test_rms_only_updates_when_asked(rng):
    ac = ActorCritic()
    obs = rng.normal(size=(8, 26)).astype(np.float32)
    ac.value(obs)
    ac.act_deterministic(obs)
    ac.act(obs, update_rms=False)
    assert ac.obs_rms.count == 0
    ac.act(obs, update_rms=True)
    assert ac.obs_rms.count == 8


def test_nonfinite_obs_do_not_poison_statistics(rng):
    ac = ActorCritic()
    obs = rng.normal(size=(4, 26)).astype(np.float32)
    obs[1, 3] = np.nan
    obs[2, 0] = np.inf
    with pytest.warns(UserWarning, match="non-finite"):
        a_env, *_ = ac.act(obs, update_rms=True)
    assert np.isfinite(ac.obs_rms.mean).all() and np.isfinite(ac.obs_rms.var).all()
    assert np.isfinite(a_env).all() and torch.isfinite(ac.last_obs_n).all()
