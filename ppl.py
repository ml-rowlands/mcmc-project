"""
ppl.py — Minimal Probabilistic Programming Language
====================================================
Built incrementally in the "Build a PPL" tab of MCMC Explorer.

Public API
----------
Distributions : Normal, HalfNormal, Gamma, Beta, Exponential, Uniform
Node classes  : Variable, Deterministic
Model         : holds the graphical model, computes joint log-prob
MCMC          : Metropolis-Hastings and HMC over a Model
Helpers       : NormalVecLikelihood
"""

import numpy as np
import scipy.stats as sp


# ── Distributions ──────────────────────────────────────────────────────────────

class Normal:
    """Gaussian distribution.  mean and std may be callables (for linking)."""
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std  = std

    def log_prob(self, x):
        mean = self.mean() if callable(self.mean) else self.mean
        std  = self.std()  if callable(self.std)  else self.std
        return float(sp.norm.logpdf(x, mean, std))

    def __repr__(self):
        return f"Normal(mean={self.mean}, std={self.std})"


class HalfNormal:
    """Half-normal distribution — used as a prior on positive quantities."""
    def __init__(self, scale=1.0):
        self.scale = scale

    def log_prob(self, x):
        if np.any(np.asarray(x) < 0):
            return -np.inf
        return float(sp.halfnorm.logpdf(x, scale=self.scale))

    def __repr__(self):
        return f"HalfNormal(scale={self.scale})"


class Gamma:
    """Gamma(alpha, beta) with *rate* parameterisation (beta = rate, scale = 1/beta)."""
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha   # shape
        self.beta  = beta    # rate

    def log_prob(self, x):
        if np.any(np.asarray(x) <= 0):
            return -np.inf
        return float(sp.gamma.logpdf(x, a=self.alpha, scale=1.0 / self.beta))

    def __repr__(self):
        return f"Gamma(alpha={self.alpha}, beta={self.beta})"


class Beta:
    """Beta(alpha, beta) — supported on (0, 1)."""
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta  = beta

    def log_prob(self, x):
        if np.any((np.asarray(x) <= 0) | (np.asarray(x) >= 1)):
            return -np.inf
        return float(sp.beta.logpdf(x, self.alpha, self.beta))

    def __repr__(self):
        return f"Beta(alpha={self.alpha}, beta={self.beta})"


class Exponential:
    """Exponential distribution with rate parameterisation (mean = 1/rate)."""
    def __init__(self, rate=1.0):
        self.rate = rate

    def log_prob(self, x):
        if np.any(np.asarray(x) < 0):
            return -np.inf
        return float(sp.expon.logpdf(x, scale=1.0 / self.rate))

    def __repr__(self):
        return f"Exponential(rate={self.rate})"


class Uniform:
    """Uniform(low, high)."""
    def __init__(self, low=0.0, high=1.0):
        self.low  = low
        self.high = high

    def log_prob(self, x):
        return float(sp.uniform.logpdf(x, self.low, self.high - self.low))

    def __repr__(self):
        return f"Uniform(low={self.low}, high={self.high})"


# ── Node classes ───────────────────────────────────────────────────────────────

class Deterministic:
    """
    A deterministic function of parent nodes.
    Evaluating it updates its value in-place for downstream nodes to read.
    """
    def __init__(self, fn):
        self.fn = fn

    def evaluate(self):
        return self.fn()


class Variable:
    """
    A single node in the probabilistic graphical model.

    Parameters
    ----------
    name          : unique string identifier
    dist          : a distribution instance (has .log_prob) or Deterministic
    parents       : list of Variable objects this node depends on
    observed      : if True, the node's value is fixed (data)
    observed_data : fixed value when observed=True
    """
    def __init__(self, name, dist, parents=None, observed=False, observed_data=None):
        self.name          = name
        self.dist          = dist
        self.parents       = parents or []
        self.observed      = observed
        self.value         = observed_data
        self.deterministic = isinstance(dist, Deterministic)


# ── Model ──────────────────────────────────────────────────────────────────────

class Model:
    """
    A probabilistic graphical model.

    Holds a dict of named Variable nodes.  Computes the joint log-probability
    by summing log_prob over all non-deterministic nodes in topological order.
    The normalising constant Z is never needed — MCMC only uses ratios.
    """

    def __init__(self):
        self.variables: dict[str, Variable] = {}

    # ── Graph construction ─────────────────────────────────────────────────

    def add_variable(self, name: str, dist, parents=None,
                     observed: bool = False, observed_data=None) -> Variable:
        if name in self.variables:
            raise ValueError(f"Variable '{name}' already exists in this model.")
        var = Variable(name, dist, parents, observed, observed_data)
        self.variables[name] = var
        return var

    def add_deterministic(self, name: str, fn, parents) -> Variable:
        if name in self.variables:
            raise ValueError(f"Variable '{name}' already exists in this model.")
        var = Variable(name, dist=Deterministic(fn), parents=parents, observed=False)
        self.variables[name] = var
        return var

    # ── Evaluation ────────────────────────────────────────────────────────

    def _topological_sort(self) -> list[Variable]:
        visited, order = set(), []

        def dfs(var):
            if var.name in visited:
                return
            visited.add(var.name)
            for p in var.parents:
                dfs(p)
            order.append(var)

        for var in self.variables.values():
            dfs(var)
        return order

    def log_prob(self, state: dict) -> float:
        """
        Evaluate the joint log-probability at *state*.

        state : dict mapping free-variable names → float values
        Returns -inf if any variable violates its domain.
        """
        logp = 0.0
        for var in self._topological_sort():
            if var.deterministic:
                var.value = var.dist.evaluate()
            else:
                if not var.observed:
                    var.value = state[var.name]
                lp = var.dist.log_prob(var.value)
                if not np.isfinite(lp):
                    return -np.inf
                logp += lp
        return logp

    @property
    def free_vars(self) -> list[str]:
        """Names of latent (non-observed, non-deterministic) variables."""
        return [name for name, var in self.variables.items()
                if not var.observed and not var.deterministic]


# ── MCMC sampler ───────────────────────────────────────────────────────────────

class MCMC:
    """
    Multivariate MCMC sampler for a Model.

    Supports two algorithms:
      'metropolis' — Random-walk Metropolis-Hastings
      'hmc'        — Hamiltonian Monte Carlo with leapfrog integration

    The chain, accepted, and proposed counters are reset on every call to
    sample(), so the object is safely reusable across multiple runs.

    Parameters
    ----------
    model         : Model instance
    initial_state : dict of starting values for each free variable
                    (defaults to 0.0 for all)
    proposal_std  : step size for Metropolis proposals (default 0.1)
    """

    def __init__(self, model: Model, initial_state: dict = None,
                 proposal_std: float = 0.1):
        self.model         = model
        self.proposal_std  = proposal_std
        self._init_state   = (
            {k: float(initial_state[k]) for k in model.free_vars}
            if initial_state is not None
            else {k: 0.0 for k in model.free_vars}
        )

    # ── Internal state ─────────────────────────────────────────────────────

    def _reset(self):
        self.current_state = dict(self._init_state)
        self.chain:  list[dict] = []
        self.accepted = 0
        self.proposed = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed > 0 else 0.0

    # ── Metropolis ─────────────────────────────────────────────────────────

    def _proposal_step(self) -> dict:
        return {k: float(np.random.normal(self.current_state[k], self.proposal_std))
                for k in self.model.free_vars}

    def _metropolis(self, n_samples: int, burn_in: int):
        for i in range(n_samples + burn_in):
            proposed   = self._proposal_step()
            lp_curr    = self.model.log_prob(self.current_state)
            lp_prop    = self.model.log_prob(proposed)

            if np.log(np.random.rand() + 1e-300) < (lp_prop - lp_curr):
                self.current_state = proposed
                if i >= burn_in:
                    self.accepted += 1

            self.proposed += 1
            if i >= burn_in:
                self.chain.append(self.current_state.copy())

    # ── HMC ────────────────────────────────────────────────────────────────

    def _grad_log_prob(self, state: dict, h: float = 1e-4) -> dict:
        """
        Central finite-difference gradient of log_prob.

        Note: Production PPLs (PyMC, Stan) use symbolic / algorithmic
        differentiation so gradients are exact and cheap.  Central finite
        differences work for any distribution class without modification.
        """
        g = {}
        for k in self.model.free_vars:
            s_p = {**state, k: state[k] + h}
            s_m = {**state, k: state[k] - h}
            g[k] = (self.model.log_prob(s_p) - self.model.log_prob(s_m)) / (2 * h)
        return g

    def _hmc(self, n_samples: int, burn_in: int,
             step_size: float = 0.05, n_leapfrog_steps: int = 20):
        """
        Hamiltonian Monte Carlo with leapfrog (velocity-Verlet) integration.

        For each iteration:
          1. Draw fresh momentum p ~ N(0, I)
          2. Run n_leapfrog_steps of leapfrog at step_size ε
          3. Accept / reject via Metropolis on ΔH
        """
        keys = self.model.free_vars

        for i in range(n_samples + burn_in):
            q = dict(self.current_state)
            p = {k: float(np.random.standard_normal()) for k in keys}

            # Half-step momentum update
            g    = self._grad_log_prob(q)
            p_hf = {k: p[k] + 0.5 * step_size * g[k] for k in keys}

            q_new, p_new = dict(q), dict(p_hf)

            for l in range(n_leapfrog_steps):
                q_new = {k: q_new[k] + step_size * p_new[k] for k in keys}
                g_new = self._grad_log_prob(q_new)
                if l < n_leapfrog_steps - 1:
                    p_new = {k: p_new[k] + step_size * g_new[k] for k in keys}
                else:
                    p_new = {k: p_new[k] + 0.5 * step_size * g_new[k] for k in keys}

            H_curr = -self.model.log_prob(q)     + 0.5 * sum(p[k]**2     for k in keys)
            H_prop = -self.model.log_prob(q_new) + 0.5 * sum(p_new[k]**2 for k in keys)

            if np.log(np.random.rand() + 1e-300) < H_curr - H_prop:
                self.current_state = q_new
                if i >= burn_in:
                    self.accepted += 1

            self.proposed += 1
            if i >= burn_in:
                self.chain.append(self.current_state.copy())

    # ── Public API ─────────────────────────────────────────────────────────

    def sample(self, method: str = 'metropolis', n_samples: int = 1000,
               burn_in: int = 500, **kwargs) -> list[dict]:
        """
        Draw samples from the model posterior.

        Parameters
        ----------
        method    : 'metropolis' or 'hmc'
        n_samples : number of post-burn-in samples to return
        burn_in   : iterations to discard at the start
        **kwargs  : passed to the chosen sampler
                    (e.g. step_size=0.05, n_leapfrog_steps=20 for HMC)

        Returns
        -------
        list of dicts, each mapping variable names → sampled values
        """
        self._reset()

        if method == 'metropolis':
            self._metropolis(n_samples, burn_in)
        elif method == 'hmc':
            kw = dict(step_size=0.05, n_leapfrog_steps=20)
            kw.update(kwargs)
            self._hmc(n_samples, burn_in, **kw)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: metropolis, hmc"
            )

        return self.chain


# ── Helpers ────────────────────────────────────────────────────────────────────

class NormalVecLikelihood:
    """
    Vectorised Normal likelihood: scores a 1-D array of observations
    against Normal(mu_var.value, sigma_var.value).

    Used as the `dist` argument of an observed Variable node when the
    likelihood depends on upstream Variable values.
    """
    def __init__(self, mu_var: Variable, sigma_var: Variable, y: np.ndarray):
        self.mu_var    = mu_var
        self.sigma_var = sigma_var
        self.y         = np.asarray(y, dtype=float)

    def log_prob(self, x):
        mu    = self.mu_var.value
        sigma = self.sigma_var.value
        if mu is None or sigma is None or np.any(np.asarray(sigma) <= 0):
            return -np.inf
        return float(np.sum(sp.norm.logpdf(self.y, mu, sigma)))
