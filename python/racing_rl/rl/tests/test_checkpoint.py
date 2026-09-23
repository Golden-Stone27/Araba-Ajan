"""DoD 8: checkpoint save → load gives identical outputs (weights + obs RMS); same seed → bit-identical 1st update."""

import math

import numpy as np
import pytest
import torch

from racing_rl.bridge.protocol import FROZEN_ENV_CONFIG_HASH, layout_hash
from racing_rl.rl import (ActorCritic, CheckpointMismatchError, PPOConfig, TargetReachVecEnv, TorchPolicy,
                          actor_critic_from_checkpoint, get_preset, load_checkpoint, restore_rng, save_checkpoint,
                          train)

CFG = PPOConfig(rollout_length=32, minibatch_size=256)
TIMING = {"time_collect_s", "time_update_s", "time_total_s", "sps"}


def _run(seed: int, ckpt_dir=None, cfg=CFG, updates=1):
    torch.manual_seed(seed)  # weights are initialised here (train() seeds again before collecting)
    env = TargetReachVecEnv(16, episode_len=20, seed=seed)
    ac = ActorCritic.from_config(cfg)
    hist = train(env, cfg, total_steps=updates * 32 * 16, ckpt_dir=ckpt_dir, seed=seed, ac=ac,
                 env_config_hash=FROZEN_ENV_CONFIG_HASH, obs_layout_hash=layout_hash())
    return hist, ac


def test_save_load_roundtrip(tmp_path, rng):
    hist, ac = _run(seed=1, ckpt_dir=tmp_path, updates=2)
    files = sorted(p.name for p in tmp_path.glob("ckpt_*.pt"))
    assert files == ["ckpt_1024.pt"]  # ckpt_every=10, so only the final update is saved
    raw = torch.load(tmp_path / files[0], map_location="cpu", weights_only=True)  # primitives + tensors only
    assert set(raw) >= {"model", "optimizer", "obs_rms", "config", "global_step", "env_config_hash",
                        "obs_layout_hash", "git_sha", "rng"}
    assert set(raw["obs_rms"]) == {"mean", "var", "count"} and set(raw["rng"]) == {"torch", "numpy"}
    assert raw["global_step"] == 1024 and raw["obs_rms"]["count"] == 1024.0
    assert raw["env_config_hash"] == FROZEN_ENV_CONFIG_HASH and raw["obs_layout_hash"] == layout_hash()

    ckpt = load_checkpoint(tmp_path / files[0], FROZEN_ENV_CONFIG_HASH, layout_hash())
    ac2 = actor_critic_from_checkpoint(ckpt)
    assert np.array_equal(ac2.obs_rms.mean, ac.obs_rms.mean) and np.array_equal(ac2.obs_rms.var, ac.obs_rms.var)
    obs = rng.normal(size=(64, 26)).astype(np.float32)
    assert np.array_equal(ac2.act_deterministic(obs), ac.act_deterministic(obs))
    assert torch.equal(ac2.value(obs), ac.value(obs))
    torch.manual_seed(5)
    s1 = ac.act_full(obs, update_rms=False)
    torch.manual_seed(5)
    s2 = ac2.act_full(obs, update_rms=False)
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(s1, s2))
    p = TorchPolicy.from_checkpoint(tmp_path / files[0], expected_env_hash=FROZEN_ENV_CONFIG_HASH)
    assert np.array_equal(p(obs), ac.act_deterministic(obs))
    assert ac.obs_rms.count == ac2.obs_rms.count == 1024  # policy calls never touch the statistics

    # optimizer state restores into a fresh Adam; rng state restores
    opt = torch.optim.Adam(ac2.parameters(), lr=1.0)
    opt.load_state_dict(ckpt["optimizer"])
    assert len(opt.state) == len(list(ac2.parameters()))
    restore_rng(ckpt, np.random.default_rng(0))

    with pytest.raises(CheckpointMismatchError):
        load_checkpoint(tmp_path / files[0], expected_env_hash="0000000000000000")
    with pytest.raises(CheckpointMismatchError):
        load_checkpoint(tmp_path / files[0], expected_obs_hash="0000000000000000")


def test_parity_architecture_roundtrip(tmp_path, rng):
    cfg = get_preset("parity")
    ac = ActorCritic.from_config(cfg)
    ac.act(rng.normal(size=(8, 26)).astype(np.float32), update_rms=True)
    path = save_checkpoint(tmp_path / "p.pt", ac, None, cfg, 0, None, None)
    ac2 = actor_critic_from_checkpoint(load_checkpoint(path))
    assert ac2.action_transform == "mlagents_scale" and isinstance(ac2.actor[1], torch.nn.SiLU)
    obs = rng.normal(size=(8, 26)).astype(np.float32)
    assert np.array_equal(ac2.act_deterministic(obs), ac.act_deterministic(obs))


def _same(a: float, b: float) -> bool:
    return a == b or (math.isnan(a) and math.isnan(b))


def test_same_seed_first_update_bit_identical():
    h1, _ = _run(seed=7)
    h2, _ = _run(seed=7)
    h3, _ = _run(seed=8)
    keys = sorted(set(h1[0]) - TIMING)
    diffs = [k for k in keys if not _same(h1[0][k], h2[0][k])]
    assert not diffs, {k: (h1[0][k], h2[0][k]) for k in diffs}
    assert h1[0]["policy_loss"] != h3[0]["policy_loss"]


def test_checkpoint_rotation(tmp_path):
    cfg = PPOConfig(rollout_length=4, minibatch_size=32)
    env = TargetReachVecEnv(8, seed=0)
    train(env, cfg, total_steps=13 * 4 * 8, ckpt_dir=tmp_path, ckpt_every=1, seed=0, max_checkpoints=10)
    steps = sorted(int(p.stem.split("_")[1]) for p in tmp_path.glob("ckpt_*.pt"))
    assert steps == [32 * k for k in range(4, 14)]
