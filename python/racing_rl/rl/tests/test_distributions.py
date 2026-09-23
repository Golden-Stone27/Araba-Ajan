"""DoD 2: log π and entropy match torch.distributions.Normal (≤ 1e-6)."""

import torch
from torch.distributions import Normal

from racing_rl.rl import ActorCritic
from racing_rl.rl import distributions as D


def test_log_prob_entropy_vs_normal_float64():
    B, A = 4096, 2
    mu = torch.randn(B, A, dtype=torch.float64)
    log_std = torch.empty(A, dtype=torch.float64).uniform_(-5, 1)
    u = mu + torch.randn(B, A, dtype=torch.float64) * 3.0
    ref = Normal(mu, torch.exp(log_std).expand(B, A))
    assert float((D.log_prob(u, mu, log_std) - ref.log_prob(u).sum(-1)).abs().max()) <= 1e-6
    ent = D.entropy(log_std, (B,))
    assert ent.shape == (B,)
    assert float((ent - ref.entropy().sum(-1)).abs().max()) <= 1e-6


def test_actor_critic_evaluate_vs_normal_float32():
    ac = ActorCritic()
    ac.log_std.data = torch.tensor([-0.3, 0.4])
    B = 1024
    obs = torch.randn(B, 26)
    u = torch.randn(B, 2)
    logp, ent, _ = ac.evaluate(obs, u)
    with torch.no_grad():
        ref = Normal(ac.actor(obs), torch.exp(ac.log_std).expand(B, 2))
        assert float((logp - ref.log_prob(u).sum(-1)).abs().max()) <= 1e-6
        assert float((ent - ref.entropy().sum(-1)).abs().max()) <= 1e-6


def test_sample_statistics():
    mu = torch.zeros(200_000, 2)
    log_std = torch.tensor([-0.5, 0.0])
    u = D.sample(mu, log_std)
    assert torch.allclose(u.std(0), torch.exp(log_std), rtol=1e-2)
    assert float(u.mean(0).abs().max()) < 1e-2
