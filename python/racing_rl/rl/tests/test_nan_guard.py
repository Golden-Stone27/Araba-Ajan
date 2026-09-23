"""Extra-a: a non-finite loss mid-update → step skipped, model/optimizer/RMS restored, lr halved, warning."""

import numpy as np
import pytest
import torch

from racing_rl.rl import ActorCritic, PPOConfig, PPOTrainer, RolloutBuffer, TargetReachVecEnv, collect_rollout


class _InfGrad(torch.autograd.Function):
    """Identity forward, infinite gradient backward."""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * float("inf")


def _filled(cfg, ac):
    env = TargetReachVecEnv(16, seed=0)
    buf = RolloutBuffer(cfg.rollout_length, 16)
    obs, _ = env.reset(seed=0)
    obs, last_v = collect_rollout(env, ac, buf, obs, cfg)
    buf.compute_gae(last_v, cfg.gamma, cfg.gae_lambda)
    return buf


@pytest.mark.parametrize("where", ["loss", "grad_norm"])
def test_nan_injection_restores_and_halves_lr(where):
    cfg = PPOConfig(rollout_length=32, minibatch_size=64)
    ac = ActorCritic.from_config(cfg)
    trainer = PPOTrainer(ac, cfg)
    buf = _filled(cfg, ac)
    m0 = trainer.update(buf, 0.0)  # a clean update first, so the optimizer has state to restore
    assert m0["nan_skipped"] == 0.0 and trainer.lr_mult == 1.0

    before = {k: v.clone() for k, v in ac.state_dict().items()}
    opt_before = {k: {n: t.clone() for n, t in s.items() if torch.is_tensor(t)}
                  for k, s in trainer.optimizer.state_dict()["state"].items()}
    rms_before = ac.obs_rms.copy()

    calls = {"n": 0}
    orig = ac.evaluate

    def poisoned(obs_n, u):
        calls["n"] += 1
        logp, ent, v = orig(obs_n, u)
        if calls["n"] == 3:  # after two real optimizer steps inside this update
            if where == "loss":
                logp = logp * torch.tensor(float("nan"))
            else:  # finite loss, infinite gradient
                v = _InfGrad.apply(v)
        return logp, ent, v

    ac.evaluate = poisoned
    with pytest.warns(RuntimeWarning, match="non-finite"):
        m = trainer.update(buf, 0.0)
    ac.evaluate = orig
    assert m["nan_skipped"] == 1.0 and m["minibatches"] == 2.0
    assert trainer.lr_mult == 0.5 and trainer.nan_skips == 1
    assert m["lr"] == pytest.approx(cfg.lr * 0.5)
    for k, v in ac.state_dict().items():
        assert torch.equal(v, before[k]), k
    for k, s in trainer.optimizer.state_dict()["state"].items():
        for n, t in opt_before[k].items():
            assert torch.equal(s[n], t), (k, n)
    assert np.array_equal(ac.obs_rms.mean, rms_before.mean) and ac.obs_rms.count == rms_before.count

    m2 = trainer.update(buf, 0.0)  # the halved lr sticks on top of the schedule
    assert m2["nan_skipped"] == 0.0 and m2["lr"] == pytest.approx(cfg.lr * 0.5)
    assert all(np.isfinite(m2[k]) for k in ("policy_loss", "value_loss", "grad_norm"))
