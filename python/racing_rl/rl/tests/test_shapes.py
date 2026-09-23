"""DoD 1: tensor shapes and dtypes (M4 table), evaluate() → [B], action ranges, shape asserts."""

import numpy as np
import pytest
import torch

from racing_rl.rl import ActorCritic, PPOConfig, RolloutBuffer, TargetReachVecEnv, collect_rollout, get_preset

T, N, O, A = 8, 5, 26, 2


def test_act_outputs(rng):
    ac = ActorCritic()
    obs = rng.uniform(-1, 1, (N, O)).astype(np.float32)
    a_env, u, logp, v = ac.act(obs, update_rms=True)
    assert a_env.shape == (N, A) and a_env.dtype == np.float32 and np.all(np.abs(a_env) <= 1.0)
    assert u.shape == (N, A) and u.dtype == torch.float32
    assert logp.shape == (N,) and v.shape == (N,) and logp.dtype == v.dtype == torch.float32
    assert ac.last_obs_n.shape == (N, O) and ac.last_obs_n.dtype == torch.float32
    assert ac.value(obs).shape == (N,)
    assert ac.act_deterministic(obs).shape == (N, A)
    assert not (u.requires_grad or logp.requires_grad or v.requires_grad)


@pytest.mark.parametrize("transform", ["clip", "mlagents_scale"])
def test_action_transform_range(rng, transform):
    ac = ActorCritic(action_transform=transform, log_std_init=1.0)  # σ = e: many samples outside [−1, 1]
    obs = rng.uniform(-1, 1, (512, O)).astype(np.float32)
    a_env, u, _, _ = ac.act(obs, update_rms=False)
    assert np.all(np.abs(a_env) <= 1.0)
    expect = np.clip(u.numpy(), -1, 1) if transform == "clip" else np.clip(u.numpy(), -3, 3) / 3
    assert np.array_equal(a_env, expect.astype(np.float32))


def test_evaluate_shapes_and_grad():
    ac = ActorCritic()
    B = 37
    logp, ent, v = ac.evaluate(torch.randn(B, O), torch.randn(B, A))
    assert logp.shape == ent.shape == v.shape == (B,)
    assert logp.requires_grad and ent.requires_grad and v.requires_grad


def test_architecture():
    ac = ActorCritic.from_config(PPOConfig())
    lin = [m for m in ac.actor if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in lin] == [(26, 256), (256, 256), (256, 2)]
    lin_c = [m for m in ac.critic if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in lin_c] == [(26, 256), (256, 256), (256, 1)]
    assert torch.equal(ac.log_std.detach(), torch.full((2,), -0.5))
    assert all(not m.bias.detach().any() for m in lin + lin_c)
    assert isinstance(ac.actor[1], torch.nn.Tanh)
    # orthogonal init: rows of the (out ≤ in) μ head are orthogonal with norm = gain 0.01
    w = lin[-1].weight.detach()
    assert torch.allclose(w @ w.T, 1e-4 * torch.eye(2), atol=1e-9)
    par = ActorCritic.from_config(get_preset("parity"))
    assert isinstance(par.actor[1], torch.nn.SiLU) and torch.equal(par.log_std.detach(), torch.zeros(2))
    ac.log_std.data.fill_(3.0)
    assert float(ac.clamped_log_std().detach().max()) == 1.0
    ac.log_std.data.fill_(-9.0)
    assert float(ac.clamped_log_std().detach().min()) == -5.0


def test_presets():
    d, p = get_preset("default"), get_preset("parity")
    assert (d.lr, d.lr_end, d.clip_eps, d.clip_eps_end, d.ent_coef, d.epochs, d.rollout_steps(16)) == \
        (3e-4, 0.0, 0.2, 0.2, 1e-3, 4, 256)
    assert (p.lr_end, p.clip_eps_end, p.ent_coef, p.ent_coef_end, p.epochs) == (1e-10, 0.1, 5e-3, 1e-5, 3)
    assert p.rollout_steps(32) == 640 and p.rollout_steps(16) == 1280
    assert PPOConfig.from_dict(p.to_dict()) == p


def test_buffer_shapes_after_rollout():
    env = TargetReachVecEnv(N, episode_len=3, seed=0)
    cfg = PPOConfig(rollout_length=T, minibatch_size=16)
    ac = ActorCritic.from_config(cfg)
    buf = RolloutBuffer(T, N)
    obs, _ = env.reset(seed=0)
    obs, last_v = collect_rollout(env, ac, buf, obs, cfg)
    buf.compute_gae(last_v, cfg.gamma, cfg.gae_lambda)
    expected = {"obs": (T, N, O), "u": (T, N, A), "logp": (T, N), "values": (T, N), "rewards": (T, N),
                "dones": (T, N), "last_values": (N,), "adv": (T, N), "returns": (T, N)}
    for name, shape in expected.items():
        t = getattr(buf, name)
        assert tuple(t.shape) == shape, name
        assert t.dtype == torch.float32, name
    assert obs.shape == (N, O) and last_v.shape == (N,)
    assert float(buf.dones.sum()) > 0  # episode_len 3 < T: truncations happened
    mbs = list(buf.minibatches(16, torch.Generator().manual_seed(0)))
    assert sum(mb.obs.shape[0] for mb in mbs) == T * N
    assert mbs[0].obs.shape == (16, O) and mbs[0].u.shape == (16, A) and mbs[0].adv.shape == (16,)


def test_buffer_shape_asserts():
    buf = RolloutBuffer(T, N)
    with pytest.raises(AssertionError):
        buf.add(torch.zeros(N, O + 1), torch.zeros(N, A), torch.zeros(N), torch.zeros(N), np.zeros(N), np.zeros(N))
    with pytest.raises(AssertionError):
        buf.add(torch.zeros(N, O), torch.zeros(N, A), torch.zeros(N, 1), torch.zeros(N), np.zeros(N), np.zeros(N))
    with pytest.raises(AssertionError):
        buf.compute_gae(torch.zeros(N), 0.99, 0.95)  # not full
    with pytest.raises(AssertionError):
        ActorCritic().act(np.zeros((N, 25), np.float32), update_rms=False)
