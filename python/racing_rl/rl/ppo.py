"""
PPOTrainer: clipped-surrogate PPO update over a filled RolloutBuffer (M4 math).

    ρ = exp(log π_θ − log π_old),  Â ← (Â − mean_mb) / (std_mb + 1e−8)
    L = −E[min(ρÂ, clip(ρ, 1±ε)Â)] + c_v · ½E[max((V−R)², (V_old + clip(V−V_old, ±ε_v) − R)²)] − c_e · H
    ‖∇‖₂ ≤ max_grad_norm;  approx_kl = E[(ρ − 1) − log ρ]

NaN guard: a non-finite loss or gradient norm aborts the update, restores the in-memory snapshot taken at its
start (model + optimizer + obs RMS), halves the learning rate for good (lr_mult) and warns.
"""

from __future__ import annotations

import copy
import math
import warnings

import torch

from .buffer import Minibatch, RolloutBuffer
from .config import LinearSchedule, PPOConfig
from .networks import ActorCritic

METRIC_KEYS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm")


class PPOTrainer:
    def __init__(self, ac: ActorCritic, cfg: PPOConfig, seed: int = 0):
        self.ac, self.cfg = ac, cfg
        self.optimizer = torch.optim.Adam(ac.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
        self.lr_schedule = LinearSchedule(cfg.lr, cfg.lr_end)
        self.clip_schedule = LinearSchedule(cfg.clip_eps, cfg.clip_eps_end)
        self.ent_schedule = LinearSchedule(cfg.ent_coef, cfg.ent_coef_end)
        self.lr_mult = 1.0  # halved by the NaN guard, on top of the schedule
        self.generator = torch.Generator().manual_seed(int(seed))  # minibatch permutations
        self.updates = 0
        self.nan_skips = 0

    # ------------------------------------------------------------------ state
    def state_dict(self) -> dict:
        return {"lr_mult": self.lr_mult, "updates": self.updates, "nan_skips": self.nan_skips,
                "generator": self.generator.get_state()}

    def load_state_dict(self, state: dict) -> None:
        self.lr_mult = float(state["lr_mult"])
        self.updates = int(state["updates"])
        self.nan_skips = int(state["nan_skips"])
        self.generator.set_state(state["generator"])

    def _snapshot(self) -> dict:
        return {"model": copy.deepcopy(self.ac.state_dict()), "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                "rms": self.ac.obs_rms.copy()}

    def _restore(self, snap: dict) -> None:
        self.ac.load_state_dict(snap["model"])
        self.optimizer.load_state_dict(snap["optimizer"])
        self.ac.obs_rms = snap["rms"].copy()

    def hyperparams(self, progress: float) -> tuple[float, float, float]:
        """(lr, ε, c_e) at progress ∈ [0, 1]."""
        return self.lr_schedule(progress) * self.lr_mult, self.clip_schedule(progress), self.ent_schedule(progress)

    # ------------------------------------------------------------------ update
    def update(self, buf: RolloutBuffer, progress: float) -> dict[str, float]:
        cfg, ac = self.cfg, self.ac
        assert buf.gae_ready, "buf.compute_gae() must run before update()"
        assert (buf.obs_dim, buf.act_dim) == (ac.obs_dim, ac.act_dim)
        torch.set_num_threads(cfg.torch_threads_update)
        progress = min(max(float(progress), 0.0), 1.0)
        lr, eps, beta = self.hyperparams(progress)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        snap = self._snapshot()
        params = list(ac.parameters())

        sums = {k: 0.0 for k in METRIC_KEYS}
        n_mb = 0
        first: dict[str, float] = {}
        early_stop = False
        epochs_done = 0
        ac.train()
        for epoch in range(cfg.epochs):
            for mb in buf.minibatches(cfg.minibatch_size, self.generator):
                loss, parts = self.compute_loss(mb, eps, beta)
                ratio, log_ratio = parts["ratio"], parts["log_ratio"]
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_frac = ((ratio - 1.0).abs() > eps).float().mean()
                if not first:
                    first = {"first_mb_ratio_maxdev": float((ratio - 1.0).abs().max()),
                             "first_mb_clip_frac": float(clip_frac), "first_mb_approx_kl": float(approx_kl)}
                if not torch.isfinite(loss):
                    return self._nan_abort(snap, "loss", epoch, n_mb, progress)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    return self._nan_abort(snap, "grad_norm", epoch, n_mb, progress)
                self.optimizer.step()

                n_mb += 1
                sums["policy_loss"] += parts["policy_loss"].item()
                sums["value_loss"] += parts["value_loss"].item()
                sums["entropy"] += parts["entropy"].item()
                sums["approx_kl"] += float(approx_kl)
                sums["clip_frac"] += float(clip_frac)
                sums["grad_norm"] += float(grad_norm)
                if cfg.target_kl is not None and float(approx_kl) > cfg.target_kl:
                    early_stop = True
                    break
            epochs_done = epoch + 1
            if early_stop:
                break

        self.updates += 1
        m = {k: s / max(n_mb, 1) for k, s in sums.items()}
        m.update(first)
        m.update(self._common(buf, lr, eps, beta))
        m.update({"nan_skipped": 0.0, "early_stop": float(early_stop), "epochs_done": float(epochs_done),
                  "minibatches": float(n_mb)})
        return m

    def compute_loss(self, mb: Minibatch, eps: float, beta: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Total loss L = L_π + c_v·L_V − c_e·H for one minibatch, plus its parts (ratio / log_ratio detached)."""
        cfg = self.cfg
        logp, ent, v = self.ac.evaluate(mb.obs, mb.u)
        log_ratio = logp - mb.logp
        ratio = torch.exp(log_ratio)
        adv = (mb.adv - mb.adv.mean()) / (mb.adv.std() + 1e-8)
        pg_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)).mean()
        v_loss_unclipped = (v - mb.returns) ** 2
        if cfg.clip_vloss:
            v_clipped = mb.values + torch.clamp(v - mb.values, -cfg.vf_clip, cfg.vf_clip)
            v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - mb.returns) ** 2).mean()
        else:
            v_loss = 0.5 * v_loss_unclipped.mean()
        entropy = ent.mean()
        loss = pg_loss + cfg.vf_coef * v_loss - beta * entropy
        return loss, {"policy_loss": pg_loss, "value_loss": v_loss, "entropy": entropy,
                      "ratio": ratio.detach(), "log_ratio": log_ratio.detach()}

    def _common(self, buf: RolloutBuffer, lr: float, eps: float, beta: float) -> dict[str, float]:
        flat = buf.flat()
        y, v = flat.returns.double(), flat.values.double()
        var_y = float(torch.var(y))
        ev = float("nan") if var_y == 0.0 else 1.0 - float(torch.var(y - v)) / var_y
        m = {"explained_var": ev, "lr": lr, "epsilon": eps, "beta": beta, "lr_mult": self.lr_mult}
        m.update(self._sigma_metrics())
        return m

    def _sigma_metrics(self) -> dict[str, float]:
        sig = self.ac.sigma()
        names = ["sigma_steer", "sigma_thr"] if len(sig) == 2 else [f"sigma_{i}" for i in range(len(sig))]
        return {n: float(s) for n, s in zip(names, sig)}

    def _nan_abort(self, snap: dict, what: str, epoch: int, n_mb: int, progress: float) -> dict[str, float]:
        self._restore(snap)
        self.optimizer.zero_grad(set_to_none=True)
        self.lr_mult *= 0.5
        self.nan_skips += 1
        lr, eps, beta = self.hyperparams(progress)
        warnings.warn(f"PPO update aborted: non-finite {what} (epoch {epoch}, after {n_mb} minibatch steps); "
                      f"model/optimizer/obs_rms restored, lr multiplier -> {self.lr_mult:g}", RuntimeWarning)
        m = {k: math.nan for k in METRIC_KEYS}
        m.update({"first_mb_ratio_maxdev": math.nan, "first_mb_clip_frac": math.nan, "first_mb_approx_kl": math.nan})
        m.update({"explained_var": math.nan, "lr": lr, "epsilon": eps, "beta": beta, "lr_mult": self.lr_mult})
        m.update(self._sigma_metrics())
        m.update({"nan_skipped": 1.0, "early_stop": 0.0, "epochs_done": float(epoch), "minibatches": float(n_mb)})
        return m
