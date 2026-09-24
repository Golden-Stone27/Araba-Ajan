"""DoD 3: GAE vs a hand-computed T = 5 example (1 terminated, 1 truncated) and vs the slow reference (≤ 1e-6)."""

import numpy as np
import torch

from racing_rl.rl import ActorCritic, PPOConfig, RolloutBuffer, collect_rollout, compute_gae, gae_reference


def test_hand_computed_t5():
    g, lam = 0.9, 0.8
    # t=1 terminated; t=3 truncated with the bootstrap already folded in: r_3 = 4 + γ·V(final_obs) = 4 + 0.9·10 = 13
    r = torch.tensor([[1.0], [2.0], [3.0], [13.0], [5.0]], dtype=torch.float64)
    v = torch.tensor([[0.5], [1.0], [1.5], [2.0], [2.5]], dtype=torch.float64)
    d = torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0]], dtype=torch.float64)
    last = torch.tensor([3.0], dtype=torch.float64)
    # δ4 = 5 + .9·3 − 2.5 = 5.2           Â4 = 5.2
    # δ3 = 13 − 2 = 11 (done)             Â3 = 11
    # δ2 = 3 + .9·2 − 1.5 = 3.3           Â2 = 3.3 + .72·11 = 11.22
    # δ1 = 2 − 1 = 1 (done)               Â1 = 1
    # δ0 = 1 + .9·1 − .5 = 1.4            Â0 = 1.4 + .72·1 = 2.12
    adv_exp = np.array([2.12, 1.0, 11.22, 11.0, 5.2])
    adv, ret = compute_gae(r, v, d, last, g, lam)
    assert np.abs(adv[:, 0].numpy() - adv_exp).max() <= 1e-6
    assert np.abs(ret[:, 0].numpy() - (adv_exp + v[:, 0].numpy())).max() <= 1e-6
    ref_adv, _ = gae_reference(r.tolist(), v.tolist(), d.tolist(), last.tolist(), g, lam)
    assert np.abs(np.array(ref_adv)[:, 0] - adv_exp).max() <= 1e-12

    # same example through the buffer (f64 compute, f32 storage)
    buf = RolloutBuffer(5, 1, 1, 1)
    for t in range(5):
        buf.add(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1), v[t], r[t], d[t])
    buf.compute_gae(last, g, lam)
    assert np.abs(buf.adv[:, 0].numpy() - adv_exp).max() <= 1e-6


def test_vectorised_vs_reference_random():
    rng = np.random.default_rng(1)
    T, N = 64, 16
    r = rng.normal(size=(T, N))
    v = rng.normal(size=(T, N)) * 3
    d = (rng.random((T, N)) < 0.1).astype(np.float64)
    last = rng.normal(size=N)
    adv, ret = compute_gae(*(torch.from_numpy(x) for x in (r, v, d, last)), 0.99, 0.95)
    ref_adv, ref_ret = gae_reference(r.tolist(), v.tolist(), d.tolist(), last.tolist(), 0.99, 0.95)
    assert np.abs(adv.numpy() - np.array(ref_adv)).max() <= 1e-6
    assert np.abs(ret.numpy() - np.array(ref_ret)).max() <= 1e-6

    # buffer path (f64 compute, f32 storage) against the reference on the f32-rounded inputs
    buf = RolloutBuffer(T, N, 1, 1)
    for t in range(T):
        buf.add(np.zeros((N, 1)), np.zeros((N, 1)), np.zeros(N), v[t], r[t], d[t])
    buf.compute_gae(last, 0.99, 0.95)
    f32 = lambda x: np.asarray(x, np.float32).astype(np.float64).tolist()  # noqa: E731
    ref_adv, _ = gae_reference(f32(r), f32(v), f32(d), f32(last), 0.99, 0.95)
    rel = np.abs(buf.adv.numpy() - np.array(ref_adv)) / np.maximum(np.abs(ref_adv), 1.0)
    assert rel.max() <= 1e-6  # only the final f32 rounding remains


class _TruncEnv:
    """2 agents; agent 0 is truncated every 2nd step, agent 1 never; r = 1; final_obs is a fixed vector."""

    num_envs = 2

    def __init__(self):
        self.t = 0
        self.final = np.full((2, 26), 0.7, np.float32)

    def step_async(self, a):
        self.t += 1

    def step_wait(self):
        trunc = np.array([self.t % 2 == 0, False])
        obs = np.full((2, 26), 0.1 * self.t, np.float32)
        return obs, np.ones(2, np.float32), np.zeros(2, bool), trunc, {"final_obs": self.final, "_final_obs": trunc}


def test_truncation_bootstrap_in_collect():
    cfg = PPOConfig(rollout_length=4, minibatch_size=4, norm_obs=False)
    ac = ActorCritic.from_config(cfg)
    buf = RolloutBuffer(4, 2)
    env = _TruncEnv()
    obs, last_v = collect_rollout(env, ac, buf, np.zeros((2, 26), np.float32), cfg)
    boot = 1.0 + cfg.gamma * float(ac.value(env.final[:1])[0])
    assert np.allclose(buf.rewards[:, 0].numpy(), [1.0, boot, 1.0, boot], atol=1e-6)
    assert np.array_equal(buf.rewards[:, 1].numpy(), np.ones(4, np.float32))
    assert np.array_equal(buf.dones.numpy(), np.array([[0, 0], [1, 0], [0, 0], [1, 0]], np.float32))
    assert torch.equal(last_v, ac.value(obs))
