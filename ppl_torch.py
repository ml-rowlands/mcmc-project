"""
ppl_torch.py — A tiny autodiff PPL using PyTorch.

The pedagogical point: every distribution's log_prob is a torch op, so the
joint log-probability is one differentiable graph.  Gradients for HMC come
from torch.autograd.grad in a single line — no finite differences.

Public API
----------
Normal, LogNormal, HalfNormal     -- distributions (log_prob returns a torch scalar)
Model                             -- flat-parameter model with a user-supplied
                                     log_prob_fn(state_dict) -> tensor
MCMC                              -- .metropolis(...) and .hmc(...)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.distributions as td


def _t(x):
    if torch.is_tensor(x):
        return x
    return torch.as_tensor(x, dtype=torch.float64)


# ── Distributions ─────────────────────────────────────────────────────────────
class Normal:
    """Normal(loc, scale).  loc/scale may be tensors or callables returning tensors."""

    def __init__(self, loc=0.0, scale=1.0):
        self.loc, self.scale = loc, scale

    def log_prob(self, x):
        loc   = self.loc()   if callable(self.loc)   else self.loc
        scale = self.scale() if callable(self.scale) else self.scale
        return td.Normal(_t(loc), _t(scale)).log_prob(_t(x)).sum()


class LogNormal:
    """LogNormal — convenient *unconstrained* prior for a positive parameter.

    If we declare  sigma ~ LogNormal(0, 1), then sigma > 0 by construction
    while the parameter we sample lives on (-inf, inf), keeping HMC happy.
    """

    def __init__(self, loc=0.0, scale=1.0):
        self.loc, self.scale = loc, scale

    def log_prob(self, x):
        return td.LogNormal(_t(self.loc), _t(self.scale)).log_prob(_t(x)).sum()


class HalfNormal:
    """HalfNormal(scale) — supported on x > 0."""

    def __init__(self, scale=1.0):
        self.scale = scale

    def log_prob(self, x):
        return td.HalfNormal(_t(self.scale)).log_prob(_t(x)).sum()


# ── Model ─────────────────────────────────────────────────────────────────────
class Model:
    """A flat-vector Bayesian model.

    Workflow:
        m = Model()
        m.add_var('mu')            # scalar
        m.add_var('u', shape=(K,)) # vector of length K
        m.set_log_prob(lambda s: ...)   # build joint logprob from s['mu'], s['u'], ...

    The flat parameter vector packs all variables in declaration order.
    """

    def __init__(self):
        self.var_names: list[str] = []
        self.var_shapes: list[tuple] = []
        self._log_prob_fn = None

    def add_var(self, name: str, shape: tuple = ()):
        self.var_names.append(name)
        self.var_shapes.append(tuple(shape))
        return self

    def set_log_prob(self, fn):
        """fn(state_dict) -> scalar torch tensor.  state_dict[name] = tensor."""
        self._log_prob_fn = fn
        return self

    @property
    def n_dim(self) -> int:
        return sum(int(np.prod(s)) if s else 1 for s in self.var_shapes)

    def unpack(self, flat: torch.Tensor) -> dict:
        out, i = {}, 0
        for n, s in zip(self.var_names, self.var_shapes):
            sz = int(np.prod(s)) if s else 1
            out[n] = flat[i : i + sz].reshape(s) if s else flat[i]
            i += sz
        return out

    def log_prob(self, flat: torch.Tensor) -> torch.Tensor:
        return self._log_prob_fn(self.unpack(flat))


# ── MCMC ──────────────────────────────────────────────────────────────────────
class MCMC:
    """Metropolis and HMC samplers driven by torch autograd."""

    def __init__(self, model: Model, init):
        self.model = model
        self.init  = torch.as_tensor(init, dtype=torch.float64)

    # The whole point of using torch: gradient in one line
    def grad_log_prob(self, q: torch.Tensor):
        q_ = q.clone().detach().requires_grad_(True)
        lp = self.model.log_prob(q_)
        (g,) = torch.autograd.grad(lp, q_)
        return lp.detach(), g.detach()

    # ── Metropolis ────────────────────────────────────────────────────────────
    def metropolis(self, n_samples: int, burn_in: int = 500,
                   proposal_std: float = 0.1, seed: int = 0):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        q = self.init.clone()
        with torch.no_grad():
            log_p = float(self.model.log_prob(q))

        chain, accepted = [], 0
        for i in range(n_samples + burn_in):
            q_star = q + torch.randn(q.shape, dtype=q.dtype) * proposal_std
            with torch.no_grad():
                log_p_star = float(self.model.log_prob(q_star))
            if np.log(rng.random()) < (log_p_star - log_p):
                q, log_p = q_star, log_p_star
                if i >= burn_in:
                    accepted += 1
            if i >= burn_in:
                chain.append(q.detach().numpy().copy())

        return np.array(chain), accepted / max(n_samples, 1)

    # ── HMC (autograd-based leapfrog) ─────────────────────────────────────────
    def hmc(self, n_samples: int, burn_in: int = 500,
            step_size: float = 0.05, n_leapfrog: int = 20, seed: int = 0):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        q = self.init.clone()
        chain, accepted = [], 0

        for i in range(n_samples + burn_in):
            p0 = torch.randn(q.shape, dtype=q.dtype)

            # Leapfrog
            q_new = q.clone()
            log_p_q, g = self.grad_log_prob(q_new)
            p_new = p0 + 0.5 * step_size * g
            for L in range(n_leapfrog):
                q_new = q_new + step_size * p_new
                log_p_qn, g = self.grad_log_prob(q_new)
                step = 0.5 if L == n_leapfrog - 1 else 1.0
                p_new = p_new + step * step_size * g

            # Accept on Hamiltonian difference
            H_curr = -float(log_p_q)  + 0.5 * float((p0    ** 2).sum())
            H_prop = -float(log_p_qn) + 0.5 * float((p_new ** 2).sum())
            if np.log(rng.random()) < (H_curr - H_prop):
                q = q_new.detach()
                if i >= burn_in:
                    accepted += 1
            if i >= burn_in:
                chain.append(q.detach().numpy().copy())

        return np.array(chain), accepted / max(n_samples, 1)
