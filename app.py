"""
MCMC Explorer — Educational Streamlit app
Covers: Metropolis-Hastings and Hamiltonian Monte Carlo (with Leapfrog integrator)
"""

import streamlit as st
import numpy as np
import scipy.stats as stats
from scipy.special import logsumexp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MCMC Explorer",
    page_icon="⛓",
    layout="wide",
)

# ── Numerical utilities ───────────────────────────────────────────────────────

def safe_log(log_target, x):
    """Evaluate log_target at x, returning -inf on any error."""
    try:
        v = float(log_target(float(x)))
        return v if np.isfinite(v) else -np.inf
    except Exception:
        return -np.inf


def grad_log(log_target, x, h=1e-5):
    """Central finite-difference gradient of log_target(x)."""
    return (safe_log(log_target, x + h) - safe_log(log_target, x - h)) / (2.0 * h)


def compute_acf(samples, max_lag=50):
    """Autocorrelation at lags 1 … max_lag."""
    n = len(samples)
    if n < max_lag + 2 or np.var(samples) < 1e-12:
        return np.zeros(max_lag)
    return np.array([
        np.corrcoef(samples[: n - k], samples[k:])[0, 1]
        for k in range(1, max_lag + 1)
    ])

# ── Target distributions ──────────────────────────────────────────────────────

def _log_normal(x):
    return float(stats.norm.logpdf(x, 0.0, 1.0))

def _log_bimodal(x):
    return float(logsumexp([
        np.log(0.4) + stats.norm.logpdf(x, -3.0, 0.7),
        np.log(0.6) + stats.norm.logpdf(x,  3.0, 1.0),
    ]))

def _log_student_t(x):
    return float(stats.t.logpdf(x, df=3))

def _log_laplace(x):
    return float(stats.laplace.logpdf(x, 0.0, 1.0))

TARGETS = {
    "Standard Normal": {
        "fn": _log_normal, "range": (-5.0, 5.0), "q0": 0.0,
        "desc": "N(0, 1) — the simplest baseline.",
    },
    "Bimodal Mixture": {
        "fn": _log_bimodal, "range": (-8.0, 8.0), "q0": 0.0,
        "desc": "0.4·N(−3, 0.7) + 0.6·N(3, 1) — tests multimodal exploration.",
    },
    "Student-t (df=3)": {
        "fn": _log_student_t, "range": (-8.0, 8.0), "q0": 0.0,
        "desc": "Heavy tails — highlights proposal mismatch.",
    },
    "Laplace": {
        "fn": _log_laplace, "range": (-7.0, 7.0), "q0": 0.0,
        "desc": "Exponential tails, non-differentiable peak.",
    },
}

# ── Metropolis-Hastings ───────────────────────────────────────────────────────

def run_metropolis(log_target, q0, n_samples, burn_in, proposal_std, seed):
    rng = np.random.default_rng(int(seed))
    q = float(q0)
    samples = np.empty(n_samples)
    n_accepted = 0
    for i in range(n_samples + burn_in):
        q_prop = float(rng.normal(q, proposal_std))
        log_alpha = safe_log(log_target, q_prop) - safe_log(log_target, q)
        if np.log(rng.random() + 1e-300) < log_alpha:
            q = q_prop
            if i >= burn_in:
                n_accepted += 1
        if i >= burn_in:
            samples[i - burn_in] = q
    return samples, n_accepted / n_samples

# ── Leapfrog integrator ───────────────────────────────────────────────────────

def leapfrog(q, p, log_target, eps, L):
    """
    Leapfrog (velocity Verlet) integrator for HMC.

    Hamiltonian:   H(q, p) = U(q) + K(p)
                           = −log π(q)  +  p²/2

    Equations of motion:
        dq/dt =  ∂H/∂p = p
        dp/dt = −∂H/∂q = ∇log π(q)

    Leapfrog scheme (L full steps of size ε):

        p ← p + (ε/2) ∇log π(q)          [half-step momentum]
        for l = 1 … L:
            q ← q + ε · p                 [full-step position]
            if l < L:
                p ← p + ε ∇log π(q)      [full-step momentum]
            else:
                p ← p + (ε/2) ∇log π(q)  [half-step on last]
        p ← −p                            [negate for time-reversal / detailed balance]

    Returns:
        q_new, p_new   — proposed state
        trajectory     — list of (q, p, H) recorded after each sub-step
    """
    U_fn = lambda q_: -safe_log(log_target, q_)
    H_fn = lambda q_, p_: U_fn(q_) + 0.5 * p_ * p_

    q, p = float(q), float(p)
    trajectory = [(q, p, H_fn(q, p))]

    p += (eps / 2.0) * grad_log(log_target, q)        # half-step momentum

    for l in range(L):
        q += eps * p                                    # full-step position
        if l < L - 1:
            p += eps * grad_log(log_target, q)          # full-step momentum
        else:
            p += (eps / 2.0) * grad_log(log_target, q) # half-step (last)
        trajectory.append((q, p, H_fn(q, p)))

    return q, -p, trajectory   # negate p for detailed balance


def euler_integration(q, p, log_target, eps, L):
    """
    Forward Euler integration of the Hamiltonian system.
    Included as a foil to leapfrog: Euler is NOT symplectic and
    energy H(q,p) drifts over time, making it unsuitable for HMC.
    """
    U_fn = lambda q_: -safe_log(log_target, q_)
    H_fn = lambda q_, p_: U_fn(q_) + 0.5 * p_ * p_
    q, p = float(q), float(p)
    trajectory = [(q, p, H_fn(q, p))]
    for _ in range(L):
        q_new = q + eps * p
        p_new = p + eps * grad_log(log_target, q)  # gradient at current q (Euler)
        q, p = q_new, p_new
        trajectory.append((q, p, H_fn(q, p)))
    return trajectory


def run_hmc(log_target, q0, n_samples, burn_in, eps, L, seed):
    rng = np.random.default_rng(int(seed))
    q = float(q0)
    samples = np.empty(n_samples)
    n_accepted = 0
    U_fn = lambda q_: -safe_log(log_target, q_)
    last_traj = None
    for i in range(n_samples + burn_in):
        p = float(rng.standard_normal())
        q_new, p_new, traj = leapfrog(q, p, log_target, eps, L)
        dH = (U_fn(q_new) + 0.5 * p_new**2) - (U_fn(q) + 0.5 * p**2)
        if np.log(rng.random() + 1e-300) < -dH:
            q = q_new
            if i >= burn_in:
                n_accepted += 1
        if i >= burn_in:
            samples[i - burn_in] = q
        last_traj = traj
    return samples, n_accepted / n_samples, last_traj

# ── Cached computation wrappers ───────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _cached_metropolis(target_name, n_samples, burn_in, proposal_std, seed):
    t = TARGETS[target_name]
    return run_metropolis(t["fn"], t["q0"], n_samples, burn_in, proposal_std, int(seed))


@st.cache_data(show_spinner=False)
def _cached_hmc(target_name, n_samples, burn_in, eps, L, seed):
    t = TARGETS[target_name]
    samples, rate, last_traj = run_hmc(t["fn"], t["q0"], n_samples, burn_in, eps, L, int(seed))
    traj_out = [list(s) for s in last_traj] if last_traj else None
    return samples, rate, traj_out


@st.cache_data(show_spinner=False)
def _cached_leapfrog_viz(target_name, q0, p0, eps, L):
    t = TARGETS[target_name]
    _, _, traj_lf = leapfrog(q0, p0, t["fn"], eps, L)
    traj_eu = euler_integration(q0, p0, t["fn"], eps, L)
    return [list(s) for s in traj_lf], [list(s) for s in traj_eu]

# ── Plotting helpers ──────────────────────────────────────────────────────────

C_BLUE   = "#636EFA"
C_RED    = "#EF553B"
C_GREEN  = "#00CC96"
C_ORANGE = "#FFA15A"
C_PURPLE = "#AB63FA"


def density_curve(log_target, x_range, n=600):
    xs = np.linspace(*x_range, n)
    lp = np.array([safe_log(log_target, x) for x in xs])
    p  = np.exp(lp - lp.max())
    dx = (x_range[1] - x_range[0]) / n
    p /= p.sum() * dx
    return xs, p


def make_diagnostics(samples, log_target, x_range, color, title=""):
    """3-panel: trace | histogram + target density | ACF."""
    acf_vals = compute_acf(samples)
    xs, pd   = density_curve(log_target, x_range)
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Trace plot", "Samples vs. target density", "Autocorrelation (ACF)"],
    )
    fig.add_trace(go.Scatter(y=samples, mode="lines",
                             line=dict(color=color, width=0.6), showlegend=False), 1, 1)
    fig.add_trace(go.Histogram(x=samples, histnorm="probability density",
                               marker_color=color, opacity=0.55, nbinsx=60,
                               showlegend=False), 1, 2)
    fig.add_trace(go.Scatter(x=xs, y=pd, mode="lines",
                             line=dict(color=C_GREEN, width=2.5),
                             name="target", showlegend=False), 1, 2)
    fig.add_trace(go.Bar(x=list(range(1, len(acf_vals) + 1)), y=acf_vals,
                         marker_color=color, showlegend=False), 1, 3)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=3)
    fig.update_layout(title=title, height=320, margin=dict(t=60, b=20, l=10, r=10))
    return fig


def make_phase_space(log_target, x_range, traj_lf, traj_eu=None, title="Phase space"):
    """H(q,p) contour map with leapfrog (and optionally Euler) trajectory."""
    qs = np.linspace(*x_range, 150)
    ps = np.linspace(-4.5, 4.5, 150)
    U_grid = np.array([[-safe_log(log_target, q) for q in qs] for _ in ps])
    _, P_grid = np.meshgrid(qs, ps)
    H_grid = U_grid + 0.5 * P_grid ** 2

    lf_q, lf_p, lf_H = zip(*traj_lf)
    H0 = lf_H[0]

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=qs, y=ps, z=H_grid,
        colorscale="Blues",
        contours=dict(start=max(0.0, H0 - 2), end=H0 + 6, size=0.4, showlabels=False),
        line_smoothing=0.85, showscale=False, opacity=0.5,
        name="H contours",
    ))
    if traj_eu is not None:
        eu_q, eu_p, _ = zip(*traj_eu)
        fig.add_trace(go.Scatter(
            x=eu_q, y=eu_p, mode="lines+markers",
            line=dict(color=C_ORANGE, width=2, dash="dash"),
            marker=dict(size=5, color=C_ORANGE),
            name="Euler",
        ))
    fig.add_trace(go.Scatter(
        x=lf_q, y=lf_p, mode="lines+markers",
        line=dict(color=C_RED, width=2.5),
        marker=dict(size=8, color=C_RED),
        name="Leapfrog",
    ))
    fig.add_trace(go.Scatter(
        x=[lf_q[0]], y=[lf_p[0]], mode="markers",
        marker=dict(size=15, color="limegreen", symbol="star"), name="Start",
    ))
    fig.add_trace(go.Scatter(
        x=[lf_q[-1]], y=[lf_p[-1]], mode="markers",
        marker=dict(size=13, color="crimson", symbol="x-thin-open",
                    line=dict(width=3, color="crimson")), name="End",
    ))
    fig.update_layout(
        title=title, xaxis_title="q  (position)", yaxis_title="p  (momentum)",
        height=440, margin=dict(t=50, b=30), legend=dict(x=0.01, y=0.99),
    )
    return fig


def make_hamiltonian_fig(traj_lf, traj_eu=None):
    """H(q,p) vs. leapfrog step — should stay ≈ flat for leapfrog, drift for Euler."""
    _, _, lf_H = zip(*traj_lf)
    H0 = lf_H[0]
    steps = list(range(len(lf_H)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=list(lf_H), mode="lines+markers",
                             line=dict(color=C_RED, width=2.5), name="Leapfrog"))
    if traj_eu is not None:
        _, _, eu_H = zip(*traj_eu)
        fig.add_trace(go.Scatter(x=steps, y=list(eu_H), mode="lines+markers",
                                 line=dict(color=C_ORANGE, width=2, dash="dash"),
                                 name="Euler"))
    fig.add_hline(y=H0, line_dash="dot", line_color="gray",
                  annotation_text="H₀  (initial energy)",
                  annotation_position="bottom right")
    fig.update_layout(
        title="Hamiltonian H(q, p) along the trajectory",
        xaxis_title="Step", yaxis_title="H(q, p)",
        height=340, margin=dict(t=50, b=30),
    )
    return fig

# ── Session state init ────────────────────────────────────────────────────────

for _k in ("metro_res", "hmc_res", "leap_viz"):
    if _k not in st.session_state:
        st.session_state[_k] = None

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_ov, tab_mh, tab_hmc, tab_cmp = st.tabs([
    "Overview", "Metropolis-Hastings", "HMC & Leapfrog", "Comparison",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_ov:
    st.title("Markov Chain Monte Carlo — Interactive Explorer")

    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown("""
### Why MCMC?

In Bayesian inference and statistical physics we often need expectations under a
distribution π(q) that we **cannot sample from directly** — yet we can evaluate
its (unnormalized) density pointwise:

```
π(q) ∝ exp( log π(q) )
```

For example, a Bayesian posterior is *prior × likelihood* up to a normalizing
constant that requires integrating over all parameter space — usually intractable.

**MCMC constructs a Markov chain whose stationary distribution is π.**
After a *burn-in* phase the chain's trajectory is approximately distributed as π,
and we use those samples to estimate means, quantiles, credible intervals, etc.

---

### Two algorithms covered in this app

| Algorithm | Core idea | Main strength | Main weakness |
|---|---|---|---|
| **Metropolis-Hastings** | Random-walk proposals + accept/reject | Simple, needs only log π | Slow mixing, random-walk scaling |
| **Hamiltonian Monte Carlo** | Gradient-guided proposals via Hamiltonian dynamics | Fast mixing, scales to high dimensions | Needs ∇log π, tuning ε and L |
        """)

    with col_r:
        st.info("""
**How to use this app**

1. **Metropolis-Hastings** tab — build intuition for the basic accept/reject loop.
2. **HMC & Leapfrog** tab — see how Hamiltonian dynamics and the leapfrog integrator power HMC, with an explicit Euler vs. leapfrog comparison.
3. **Comparison** tab — run both side-by-side on the same target and compare ACF / acceptance rates.
        """)
        st.markdown("---")
        st.markdown("""
**Reading the diagnostics**

| Plot | What to look for |
|---|---|
| Trace | Should look like "noise around a mean" with no trend |
| Histogram | Should match the green target density |
| ACF | Fast decay toward 0 → good mixing |
| Acceptance rate | Metropolis: aim 20–50 %; HMC: aim 60–90 % |
        """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — METROPOLIS-HASTINGS
# ─────────────────────────────────────────────────────────────────────────────
with tab_mh:
    st.header("Metropolis-Hastings Algorithm")

    with st.expander("Algorithm & intuition", expanded=True):
        c_alg, c_eq = st.columns(2)
        with c_alg:
            st.markdown("**Pseudocode**")
            st.code("""\
Initialize q ← q₀

for i = 1, 2, …, burn_in + n_samples:

    # 1. Propose a new state
    q*  ~  N(q, σ²)            # Gaussian centred at current q

    # 2. Metropolis acceptance ratio
    log α = log π(q*) − log π(q)

    # 3. Accept or reject
    u ~ Uniform(0, 1)
    if log u < log α:
        q ← q*                 # accept proposal
    else:
        q ← q                  # stay at current state

    if i > burn_in:
        record q               # collect sample
""", language="text")

        with c_eq:
            st.markdown("**Key properties**")
            st.markdown("""
The Gaussian proposal is *symmetric*: q* | q has the same density as q | q*.
That symmetry cancels the Hastings correction, leaving only the target ratio.
""")
            st.latex(r"\alpha = \min\!\left(1,\;\frac{\pi(q^*)}{\pi(q)}\right)")
            st.markdown("In log-space (numerically stable):")
            st.latex(r"\log\alpha = \log\pi(q^*) - \log\pi(q)")
            st.markdown("""
**What the chain is doing**

The chain takes a random walk through parameter space.
Uphill moves (higher density) are *always* accepted.
Downhill moves are accepted *probabilistically* in proportion to the density ratio.
This guarantees the chain's stationary distribution is π.
            """)
            st.info("""
**Burn-in** discards the initial transient while the chain converges to π.
The remaining samples are used for inference.
            """)

    st.markdown("---")

    col_ctrl, col_plots = st.columns([1, 3])

    with col_ctrl:
        st.subheader("Controls")
        mh_tgt   = st.selectbox("Target distribution", list(TARGETS.keys()), key="mh_tgt")
        mh_std   = st.slider("Proposal σ", 0.05, 5.0, 1.0, 0.05, key="mh_std")
        mh_n     = st.slider("Samples (after burn-in)", 500, 10_000, 3_000, 500, key="mh_n")
        mh_burn  = st.slider("Burn-in", 100, 2_000, 500, 100, key="mh_burn")
        mh_seed  = st.number_input("Random seed", 0, 9_999, 42, key="mh_seed")
        run_mh   = st.button("Run Metropolis", type="primary", key="run_mh_btn")

        st.markdown(f"*{TARGETS[mh_tgt]['desc']}*")
        st.markdown("---")
        st.markdown("""
**Proposal width effects**

| σ | Behavior |
|---|---|
| Too small | Tiny steps, near-100 % acceptance, very slow mixing |
| ~Optimal | 20–50 % acceptance, good mixing |
| Too large | Huge jumps, near-0 % acceptance, chain gets stuck |

Drag the σ slider to observe each regime.
        """)

    if run_mh:
        t = TARGETS[mh_tgt]
        with st.spinner("Running Metropolis-Hastings…"):
            samps, rate = _cached_metropolis(mh_tgt, mh_n, mh_burn, mh_std, int(mh_seed))
        st.session_state["metro_res"] = {
            "samples": samps, "rate": rate,
            "target": mh_tgt, "n": mh_n, "std": mh_std,
        }

    with col_plots:
        r = st.session_state["metro_res"]
        if r is not None:
            t = TARGETS[r["target"]]

            m1, m2, m3 = st.columns(3)
            m1.metric("Acceptance rate", f"{r['rate']:.1%}")
            m2.metric("Samples collected", f"{r['n']:,}")
            m3.metric("Proposal σ", f"{r['std']:.2f}")

            fig_diag = make_diagnostics(
                r["samples"], t["fn"], t["range"], C_BLUE,
                f"Metropolis-Hastings on {r['target']}",
            )
            st.plotly_chart(fig_diag, use_container_width=True)

            # Step-size comparison
            st.subheader("Step-size comparison")
            st.markdown(
                "Three proposal widths on the same target — notice how the trace and ACF change:"
            )
            std_vals   = [0.1, mh_std, 5.0]
            std_labels = ["σ = 0.1  (too small)", f"σ = {mh_std}  (selected)", "σ = 5.0  (too large)"]
            colors_cmp = [C_PURPLE, C_BLUE, C_ORANGE]

            fig_cmp = make_subplots(rows=2, cols=3,
                                    subplot_titles=std_labels,
                                    row_heights=[0.55, 0.45])
            for ci, (sv, col) in enumerate(zip(std_vals, colors_cmp), start=1):
                s2, r2 = _cached_metropolis(r["target"], 2_000, 500, sv, 42)
                a2     = compute_acf(s2, max_lag=30)
                # Trace
                fig_cmp.add_trace(
                    go.Scatter(y=s2, mode="lines", line=dict(color=col, width=0.7),
                               showlegend=False,
                               name=f"accept={r2:.0%}"), 1, ci,
                )
                fig_cmp.add_annotation(
                    text=f"accept = {r2:.0%}", showarrow=False,
                    x=0.5, y=1.0, xanchor="center", yanchor="top",
                    xref=f"x{ci} domain", yref=f"y{ci} domain",
                    font=dict(size=12, color=col),
                )
                # ACF
                fig_cmp.add_trace(
                    go.Bar(x=list(range(1, len(a2)+1)), y=a2,
                           marker_color=col, showlegend=False), 2, ci,
                )
            fig_cmp.add_hline(y=0, line_dash="dash", line_color="gray", row=2)
            fig_cmp.update_layout(height=380, margin=dict(t=60, b=20))
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.info("Configure parameters above and click **Run Metropolis** to see results.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — HMC & LEAPFROG
# ─────────────────────────────────────────────────────────────────────────────
with tab_hmc:
    st.header("Hamiltonian Monte Carlo & the Leapfrog Integrator")

    # ── Hamiltonian mechanics ──────────────────────────────────────────────────
    with st.expander("Hamiltonian mechanics & the HMC idea", expanded=True):
        h_l, h_r = st.columns(2)
        with h_l:
            st.markdown("""
**Augmenting with momentum**

HMC introduces an auxiliary *momentum* variable p sampled fresh each iteration:

```
p  ~  N(0, 1)
```

The joint distribution of position q and momentum p is:

```
π(q, p) ∝ exp( −H(q, p) )
```

Marginalizing over p recovers the original target π(q).
The **Hamiltonian** H splits into potential and kinetic energy:
""")
            st.latex(r"""
H(q,\,p) \;=\; \underbrace{-\log\pi(q)}_{U(q)\;\text{(potential)}}
               \;+\; \underbrace{\tfrac{p^2}{2}}_{K(p)\;\text{(kinetic)}}
""")
            st.markdown("""
**Why this helps**

Hamiltonian dynamics conserve H, so the system evolves along
constant-energy surfaces — it explores the target *without random-walk diffusion*.
The gradient ∇log π(q) acts as a restoring force toward high-density regions.

An HMC step runs these dynamics forward for time T = ε·L, then accepts or rejects
via the usual Metropolis criterion.  Because H is (approximately) conserved by the
leapfrog integrator, the acceptance rate is typically very high.
""")
        with h_r:
            st.markdown("**Equations of motion**")
            st.latex(r"""
\frac{dq}{dt} = \frac{\partial H}{\partial p} = p
\qquad
\frac{dp}{dt} = -\frac{\partial H}{\partial q} = \nabla\!\log\pi(q)
""")
            st.markdown("**One HMC iteration**")
            st.code("""\
1. Sample fresh momentum:
       p  ~  N(0, 1)

2. Run leapfrog for L steps of size ε
       (q, p)  →  (q*, p*)

3. Metropolis accept/reject:
       ΔH = H(q*, p*) − H(q, p)
       accept with prob  min(1, exp(−ΔH))

4. If rejected: keep current q
""", language="text")
            st.info("""
**Key insight:** if the leapfrog conserves H perfectly, ΔH = 0 and every
proposal is accepted.  The leapfrog is not exact, but its energy error is
O(ε²) per step — small enough to maintain high acceptance.
            """)

    st.markdown("---")

    # ── Leapfrog explanation ────────────────────────────────────────────────────
    st.subheader("The Leapfrog Integrator")

    with st.expander("Leapfrog vs. Euler — and why it matters for HMC", expanded=True):
        lf_l, lf_r = st.columns(2)
        with lf_l:
            st.markdown("**Leapfrog (velocity Verlet)**")
            st.markdown("""
The leapfrog staggers position and momentum updates at **half-integer time steps**,
which is the same scheme you may have seen as the *velocity Verlet* method in
numerical ODEs:
""")
            st.latex(r"""
\begin{aligned}
p_{1/2}  &\leftarrow p_0     + \tfrac{\varepsilon}{2}\,\nabla\!\log\pi(q_0) \\[4pt]
q_1      &\leftarrow q_0     + \varepsilon\, p_{1/2} \\[4pt]
p_1      &\leftarrow p_{1/2} + \tfrac{\varepsilon}{2}\,\nabla\!\log\pi(q_1)
\end{aligned}
""")
            st.markdown("""
For L > 1 steps the adjacent half-steps fuse into full steps:
""")
            st.code("""\
leapfrog(q, p, ε, L):
    p  +=  (ε/2) · ∇log π(q)        # half-step momentum

    for l = 1 to L:
        q  +=  ε · p                 # full-step position
        if l < L:
            p  +=  ε · ∇log π(q)    # full-step momentum
        else:
            p  +=  (ε/2) · ∇log π(q) # half-step (last)

    p  ←  −p                         # time-reversal for detailed balance
""", language="text")

        with lf_r:
            st.markdown("**Why not forward Euler?**")
            st.markdown("""
Forward Euler applied to Hamilton's equations:
```
q_{t+1} = q_t + ε · p_t
p_{t+1} = p_t + ε · ∇log π(q_t)
```
is **not symplectic** — it does not preserve a conserved quantity.

As a result, H(q, p) grows over time (energy is injected into the system).
In an HMC context, this means:

- Large ΔH → acceptance probability ≈ 0
- The Metropolis step cannot rescue a badly drifting integrator
- Long trajectories (large L) become unusable

**Leapfrog's advantage:**

- Energy error is O(ε²) per step and **bounded** over all time
- Euler's error is O(ε) and **accumulates without bound**
- This is the same reason symplectic integrators are preferred for long-time
  integration in orbital mechanics and molecular dynamics
""")
            st.info("""
**Connection to your DE coursework**

The leapfrog is a 2nd-order **symplectic** integrator.  It exactly preserves
a *modified* Hamiltonian H̃ = H + O(ε²), which is why energy stays bounded
even over very long integration times — something Euler and even standard
Runge-Kutta cannot guarantee for Hamiltonian systems.
            """)

    st.markdown("---")

    # ── Interactive leapfrog visualizer ────────────────────────────────────────
    st.subheader("Leapfrog visualizer — phase space & energy conservation")

    lv_l, lv_r = st.columns([1, 3])
    with lv_l:
        lv_tgt  = st.selectbox("Target", list(TARGETS.keys()), key="lv_tgt",
                               help="Standard Normal gives circular orbits, easiest to read")
        lv_q0   = st.slider("Initial position q₀", -3.0, 3.0, 1.5, 0.1, key="lv_q0")
        lv_p0   = st.slider("Initial momentum p₀", -3.0, 3.0, 1.0, 0.1, key="lv_p0")
        lv_eps  = st.slider("Step size ε", 0.02, 0.8, 0.2, 0.02, key="lv_eps")
        lv_L    = st.slider("Steps L", 1, 80, 25, 1, key="lv_L")
        show_eu = st.checkbox("Show Euler trajectory (for comparison)", value=True)
        run_lv  = st.button("Visualize trajectory", type="primary", key="run_lv_btn")

        st.markdown("---")
        st.markdown("""
**What to look for**

- **Phase space:** for Standard Normal the constant-H surfaces are circles.
  Leapfrog stays on the circle; Euler spirals outward.
- **H plot:** leapfrog stays near H₀; Euler drifts upward.
- Try increasing ε to see both integrators degrade — but leapfrog degrades
  much more gracefully.
        """)

    if run_lv:
        traj_lf_raw, traj_eu_raw = _cached_leapfrog_viz(
            lv_tgt, lv_q0, lv_p0, lv_eps, lv_L,
        )
        st.session_state["leap_viz"] = {
            "lf": traj_lf_raw, "eu": traj_eu_raw,
            "target": lv_tgt, "show_eu": show_eu,
        }

    with lv_r:
        lv = st.session_state["leap_viz"]
        if lv is not None:
            t_lv   = TARGETS[lv["target"]]
            traj_lf = lv["lf"]
            traj_eu = lv["eu"] if show_eu else None

            pc1, pc2 = st.columns(2)
            with pc1:
                st.plotly_chart(
                    make_phase_space(t_lv["fn"], t_lv["range"],
                                     traj_lf, traj_eu, "Phase space (q, p)"),
                    use_container_width=True,
                )
            with pc2:
                st.plotly_chart(
                    make_hamiltonian_fig(traj_lf, traj_eu),
                    use_container_width=True,
                )

            # ΔH metrics
            H0   = traj_lf[0][2]
            dH_lf = abs(traj_lf[-1][2] - H0)
            mc1, mc2 = st.columns(2)
            mc1.metric("Leapfrog  |ΔH|", f"{dH_lf:.6f}")
            if traj_eu is not None:
                dH_eu = abs(traj_eu[-1][2] - H0)
                mc2.metric("Euler  |ΔH|", f"{dH_eu:.6f}",
                           delta=f"{dH_eu - dH_lf:+.6f} vs leapfrog",
                           delta_color="inverse")
        else:
            st.info("Set parameters and click **Visualize trajectory**.")

    st.markdown("---")

    # ── Full HMC sampler ────────────────────────────────────────────────────────
    st.subheader("Full HMC sampler")

    hmc_l, hmc_r = st.columns([1, 3])
    with hmc_l:
        st.subheader("Controls")
        hmc_tgt  = st.selectbox("Target", list(TARGETS.keys()), key="hmc_tgt")
        hmc_eps  = st.slider("Step size ε", 0.02, 0.8, 0.2, 0.02, key="hmc_eps")
        hmc_L    = st.slider("Leapfrog steps L", 1, 50, 20, 1, key="hmc_L")
        hmc_n    = st.slider("Samples", 500, 10_000, 3_000, 500, key="hmc_n")
        hmc_burn = st.slider("Burn-in", 100, 2_000, 500, 100, key="hmc_burn")
        hmc_seed = st.number_input("Random seed", 0, 9_999, 42, key="hmc_seed")
        run_hmc_btn = st.button("Run HMC", type="primary", key="run_hmc_btn")

        st.markdown(f"*{TARGETS[hmc_tgt]['desc']}*")
        st.markdown("---")
        st.markdown("""
**Tuning guide**

- Start with ε = 0.1–0.3 and L = 10–30
- Target 60–90 % acceptance rate
- Larger L → lower ACF but more gradient evaluations per sample
- Too large ε → big ΔH → many rejections
- Bimodal target: crossing the barrier between modes requires
  enough kinetic energy (p ~ N(0,1) gives |p| > 2.9 only ~0.4% of the time)
        """)

    if run_hmc_btn:
        with st.spinner("Running HMC…"):
            samps_h, rate_h, last_t = _cached_hmc(
                hmc_tgt, hmc_n, hmc_burn, hmc_eps, hmc_L, int(hmc_seed),
            )
        st.session_state["hmc_res"] = {
            "samples": samps_h, "rate": rate_h, "target": hmc_tgt,
            "n": hmc_n, "eps": hmc_eps, "L": hmc_L, "last_traj": last_t,
        }

    with hmc_r:
        rh = st.session_state["hmc_res"]
        if rh is not None:
            t_h = TARGETS[rh["target"]]
            h1, h2, h3 = st.columns(3)
            h1.metric("Acceptance rate", f"{rh['rate']:.1%}")
            h2.metric("Samples", f"{rh['n']:,}")
            h3.metric("Trajectory length ε·L", f"{rh['eps'] * rh['L']:.2f}")

            fig_hd = make_diagnostics(
                rh["samples"], t_h["fn"], t_h["range"], C_RED,
                f"HMC on {rh['target']}",
            )
            st.plotly_chart(fig_hd, use_container_width=True)

            if rh["last_traj"] is not None:
                st.markdown("**Phase-space trajectory (last HMC step)**")
                fig_lt = make_phase_space(
                    t_h["fn"], t_h["range"],
                    rh["last_traj"],
                    title="Last leapfrog trajectory",
                )
                st.plotly_chart(fig_lt, use_container_width=True)
        else:
            st.info("Configure parameters and click **Run HMC** to see results.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tab_cmp:
    st.header("Metropolis-Hastings vs. HMC — side-by-side")
    st.markdown("""
Run both algorithms on the same target with the same number of samples.
The **ACF overlay** at the bottom is the clearest signal: faster decay → better mixing
→ fewer samples required for the same estimation quality.
    """)

    cc_l, cc_r = st.columns([1, 3])
    with cc_l:
        cmp_tgt  = st.selectbox("Target", list(TARGETS.keys()), key="cmp_tgt")
        cmp_n    = st.slider("Samples each", 500, 8_000, 3_000, 500, key="cmp_n")
        cmp_burn = st.slider("Burn-in each", 100, 2_000, 500, 100, key="cmp_burn")
        cmp_seed = st.number_input("Random seed", 0, 9_999, 42, key="cmp_seed")
        st.markdown("**Metropolis settings**")
        cmp_std  = st.slider("Proposal σ", 0.05, 5.0, 1.0, 0.05, key="cmp_std")
        st.markdown("**HMC settings**")
        cmp_eps  = st.slider("Step size ε", 0.02, 0.8, 0.2, 0.02, key="cmp_eps")
        cmp_L    = st.slider("Leapfrog steps L", 1, 50, 20, 1, key="cmp_L")
        run_cmp  = st.button("Run comparison", type="primary", key="run_cmp_btn")

    with cc_r:
        if run_cmp:
            t_c = TARGETS[cmp_tgt]
            with st.spinner("Running both samplers…"):
                mh_s, mh_r = _cached_metropolis(
                    cmp_tgt, cmp_n, cmp_burn, cmp_std, int(cmp_seed),
                )
                hmc_s, hmc_r, _ = _cached_hmc(
                    cmp_tgt, cmp_n, cmp_burn, cmp_eps, cmp_L, int(cmp_seed),
                )

            acf_mh  = compute_acf(mh_s)
            acf_hmc = compute_acf(hmc_s)

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("MH acceptance",   f"{mh_r:.1%}")
            r2.metric("HMC acceptance",  f"{hmc_r:.1%}")
            r3.metric("MH ACF lag-1",    f"{acf_mh[0]:.3f}")
            r4.metric("HMC ACF lag-1",   f"{acf_hmc[0]:.3f}",
                      delta=f"{acf_hmc[0] - acf_mh[0]:+.3f} vs MH",
                      delta_color="inverse")

            fig_mh_d = make_diagnostics(mh_s, t_c["fn"], t_c["range"], C_BLUE,
                                        f"Metropolis — {cmp_tgt}")
            fig_hmc_d = make_diagnostics(hmc_s, t_c["fn"], t_c["range"], C_RED,
                                         f"HMC — {cmp_tgt}")
            st.plotly_chart(fig_mh_d,  use_container_width=True)
            st.plotly_chart(fig_hmc_d, use_container_width=True)

            # Combined ACF comparison
            fig_acf = go.Figure()
            fig_acf.add_trace(go.Scatter(
                x=list(range(1, len(acf_mh)+1)), y=acf_mh,
                mode="lines", name="Metropolis", line=dict(color=C_BLUE, width=2.5),
            ))
            fig_acf.add_trace(go.Scatter(
                x=list(range(1, len(acf_hmc)+1)), y=acf_hmc,
                mode="lines", name="HMC", line=dict(color=C_RED, width=2.5),
            ))
            fig_acf.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_acf.update_layout(
                title="ACF comparison — faster decay to 0 means better mixing",
                xaxis_title="Lag", yaxis_title="Autocorrelation",
                height=300, margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig_acf, use_container_width=True)
        else:
            st.info("Configure both samplers and click **Run comparison**.")
