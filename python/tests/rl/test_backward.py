"""DoD 4: backward on a synthetic batch (B = 1024): finite loss, finite non-zero grads, params move, ‖∇‖ ≤ 0.5."""

import torch

from racing_rl.rl import ActorCritic, Minibatch, PPOConfig, PPOTrainer


def test_backward_synthetic_batch():
    cfg = PPOConfig()
    ac = ActorCritic.from_config(cfg)
    trainer = PPOTrainer(ac, cfg)
    B = 1024
    obs, u = torch.randn(B, 26), torch.randn(B, 2)
    with torch.no_grad():
        logp_old, _, v_old = ac.evaluate(obs, u)
    mb = Minibatch(obs, u, logp_old + 0.05 * torch.randn(B), v_old + 0.1 * torch.randn(B), torch.randn(B),
                   torch.randn(B))

    loss, parts = trainer.compute_loss(mb, cfg.clip_eps, cfg.ent_coef)
    assert torch.isfinite(loss)
    trainer.optimizer.zero_grad()
    loss.backward()
    groups = {"actor": list(ac.actor.parameters()), "critic": list(ac.critic.parameters()), "log_std": [ac.log_std]}
    for name, params in groups.items():
        for p in params:
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name
            assert float(p.grad.abs().sum()) > 0.0, name
    before = {k: v.detach().clone() for k, v in ac.state_dict().items()}
    pre = torch.nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
    post = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in ac.parameters()]))
    assert torch.isfinite(pre)
    assert float(post) <= cfg.max_grad_norm + 1e-6
    trainer.optimizer.step()
    for k, v in ac.state_dict().items():
        assert not torch.equal(v, before[k]), f"{k} did not change"
