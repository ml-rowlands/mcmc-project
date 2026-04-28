"""
MCMC Explorer — step-by-step interactive educational app
Metropolis-Hastings and Hamiltonian Monte Carlo
"""

import streamlit as st
import numpy as np
import scipy.stats as stats
from scipy.special import logsumexp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os, io, warnings

import ppl as _ppl

st.set_page_config(page_title="MCMC Explorer", page_icon="⛓", layout="wide")

# ── Numerical utilities ───────────────────────────────────────────────────────

def safe_log(log_target, x):
    try:
        v = float(log_target(float(x)))
        return v if np.isfinite(v) else -np.inf
    except Exception:
        return -np.inf

def grad_log(log_target, x, h=1e-5):
    return (safe_log(log_target, x + h) - safe_log(log_target, x - h)) / (2.0 * h)

def compute_acf(samples, max_lag=50):
    n = len(samples)
    if n < max_lag + 2 or np.var(samples) < 1e-12:
        return np.zeros(max_lag)
    return np.array([np.corrcoef(samples[:n-k], samples[k:])[0, 1]
                     for k in range(1, max_lag + 1)])

# ── Target distributions ──────────────────────────────────────────────────────

def _log_normal(x):    return float(stats.norm.logpdf(x, 0.0, 1.0))
def _log_bimodal(x):
    return float(logsumexp([np.log(0.4) + stats.norm.logpdf(x, -3.0, 0.7),
                             np.log(0.6) + stats.norm.logpdf(x,  3.0, 1.0)]))
def _log_student_t(x): return float(stats.t.logpdf(x, df=3))
def _log_laplace(x):   return float(stats.laplace.logpdf(x, 0.0, 1.0))

TARGETS = {
    "Standard Normal":  {"fn": _log_normal,    "range": (-5., 5.),  "q0": 0., "desc": "N(0,1)"},
    "Bimodal Mixture":  {"fn": _log_bimodal,   "range": (-8., 8.),  "q0": 0., "desc": "0.4·N(−3,0.7)+0.6·N(3,1)"},
    "Student-t (df=3)": {"fn": _log_student_t, "range": (-8., 8.),  "q0": 0., "desc": "Heavy tails"},
    "Laplace":          {"fn": _log_laplace,   "range": (-7., 7.),  "q0": 0., "desc": "Exponential tails"},
}

# ── Metropolis ────────────────────────────────────────────────────────────────

def run_metropolis(log_target, q0, n_samples, burn_in, proposal_std, seed):
    rng = np.random.default_rng(int(seed))
    q, samples, n_acc = float(q0), np.empty(n_samples), 0
    for i in range(n_samples + burn_in):
        qp = float(rng.normal(q, proposal_std))
        if np.log(rng.random() + 1e-300) < safe_log(log_target, qp) - safe_log(log_target, q):
            q = qp
            if i >= burn_in: n_acc += 1
        if i >= burn_in: samples[i - burn_in] = q
    return samples, n_acc / n_samples

def run_metro_chain(log_target, q0, n_steps, proposal_std, seed):
    """Return list of (q_current, q_proposed, accepted) for each step."""
    rng = np.random.default_rng(int(seed))
    q = float(q0)
    steps = []
    for _ in range(n_steps):
        qp = float(rng.normal(q, proposal_std))
        acc = bool(np.log(rng.random() + 1e-300) < safe_log(log_target, qp) - safe_log(log_target, q))
        steps.append((q, qp, acc))
        if acc: q = qp
    return steps

# ── Leapfrog + HMC ────────────────────────────────────────────────────────────

def leapfrog(q, p, log_target, eps, L):
    U = lambda q_: -safe_log(log_target, q_)
    H = lambda q_, p_: U(q_) + 0.5 * p_ * p_
    q, p = float(q), float(p)
    traj = [(q, p, H(q, p))]
    p += (eps / 2.0) * grad_log(log_target, q)
    for l in range(L):
        q += eps * p
        p += (eps if l < L - 1 else eps / 2.0) * grad_log(log_target, q)
        traj.append((q, p, H(q, p)))
    return q, -p, traj

def euler_integration(q, p, log_target, eps, L):
    U = lambda q_: -safe_log(log_target, q_)
    H = lambda q_, p_: U(q_) + 0.5 * p_ * p_
    q, p = float(q), float(p)
    traj = [(q, p, H(q, p))]
    for _ in range(L):
        q_new = q + eps * p
        p_new = p + eps * grad_log(log_target, q)
        q, p = q_new, p_new
        traj.append((q, p, H(q, p)))
    return traj

def run_hmc(log_target, q0, n_samples, burn_in, eps, L, seed):
    rng = np.random.default_rng(int(seed))
    q, samples, n_acc = float(q0), np.empty(n_samples), 0
    U = lambda q_: -safe_log(log_target, q_)
    last_traj = None
    for i in range(n_samples + burn_in):
        p = float(rng.standard_normal())
        q_new, p_new, traj = leapfrog(q, p, log_target, eps, L)
        dH = (U(q_new) + 0.5*p_new**2) - (U(q) + 0.5*p**2)
        if np.log(rng.random() + 1e-300) < -dH:
            q = q_new
            if i >= burn_in: n_acc += 1
        if i >= burn_in: samples[i - burn_in] = q
        last_traj = traj
    return samples, n_acc / n_samples, last_traj

# ── Cached wrappers ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _cached_metro(target_name, n, burn, std, seed):
    t = TARGETS[target_name]
    return run_metropolis(t["fn"], t["q0"], n, burn, std, int(seed))

@st.cache_data(show_spinner=False)
def _cached_hmc(target_name, n, burn, eps, L, seed):
    t = TARGETS[target_name]
    s, r, lt = run_hmc(t["fn"], t["q0"], n, burn, eps, L, int(seed))
    return s, r, ([list(x) for x in lt] if lt else None)

# ── Animation builders ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_metro_anim(target_name, n_steps, proposal_std, seed):
    """
    Plotly animation of Metropolis chain exploring the posterior.
    Each proposal step is two sub-frames:
      A) current (blue) + proposal (orange) visible
      B) result: accepted (green ball moves) or rejected (red ball stays)
    """
    t = TARGETS[target_name]
    fn, xr = t["fn"], t["range"]

    xs = np.linspace(*xr, 300)
    lps = np.array([safe_log(fn, x) for x in xs])
    pd = np.exp(lps - lps.max())
    pd /= pd.sum() * (xr[1] - xr[0]) / 300

    def y_at(x):
        return float(np.interp(x, xs, pd))

    chain_steps = run_metro_chain(fn, t["q0"], n_steps, proposal_std, int(seed))
    chain = [t["q0"]]
    frames = []

    for i, (q_curr, q_prop, accepted) in enumerate(chain_steps):
        trail_x, trail_y = chain[:], [y_at(x) for x in chain]

        # Frame A: show current + proposal
        frames.append(go.Frame(
            name=f"{i}a",
            traces=[2, 3, 4, 5],
            data=[
                go.Scatter(x=trail_x, y=trail_y, mode="markers",
                           marker=dict(size=6, color="#636EFA", opacity=0.25)),
                go.Scatter(x=[q_curr], y=[y_at(q_curr)], mode="markers",
                           marker=dict(size=22, color="#636EFA",
                                       line=dict(width=2.5, color="#2929cc"))),
                go.Scatter(x=[q_prop], y=[y_at(q_prop)], mode="markers",
                           marker=dict(size=18, color="#FFA15A", opacity=0.95,
                                       line=dict(width=2, color="#cc7700"))),
                go.Scatter(x=chain[:], y=[-0.025] * len(chain), mode="markers",
                           marker=dict(size=4, color="#636EFA", symbol="line-ns-open",
                                       line=dict(width=1.5, color="#636EFA"))),
            ],
        ))

        chain.append(q_prop if accepted else q_curr)
        new_q = chain[-1]
        ball_col = "#00CC96" if accepted else "#EF553B"
        border   = "#006644" if accepted else "#991100"

        # Frame B: show result
        frames.append(go.Frame(
            name=f"{i}b",
            traces=[2, 3, 4, 5],
            data=[
                go.Scatter(x=chain[:-1], y=[y_at(x) for x in chain[:-1]], mode="markers",
                           marker=dict(size=6, color="#636EFA", opacity=0.25)),
                go.Scatter(x=[new_q], y=[y_at(new_q)], mode="markers",
                           marker=dict(size=22, color=ball_col,
                                       line=dict(width=2.5, color=border))),
                go.Scatter(x=[], y=[], mode="markers",
                           marker=dict(size=1, color="rgba(0,0,0,0)")),
                go.Scatter(x=chain[:], y=[-0.025] * len(chain), mode="markers",
                           marker=dict(size=4, color="#636EFA", symbol="line-ns-open",
                                       line=dict(width=1.5, color="#636EFA"))),
            ],
        ))

    fig = go.Figure(
        data=[
            go.Scatter(x=xs, y=pd, fill="tozeroy", mode="none",
                       fillcolor="rgba(0,204,150,0.12)", showlegend=False),
            go.Scatter(x=xs, y=pd, mode="lines",
                       line=dict(color="#00CC96", width=2.5), name="π(q)"),
            go.Scatter(x=[], y=[], mode="markers",
                       marker=dict(size=6, color="#636EFA", opacity=0.25), name="Past states"),
            go.Scatter(x=[t["q0"]], y=[y_at(t["q0"])], mode="markers",
                       marker=dict(size=22, color="#636EFA",
                                   line=dict(width=2.5, color="#2929cc")), name="Current q"),
            go.Scatter(x=[], y=[], mode="markers",
                       marker=dict(size=18, color="#FFA15A", opacity=0.0), name="Proposal q*"),
            go.Scatter(x=[t["q0"]], y=[-0.025], mode="markers",
                       marker=dict(size=4, color="#636EFA", symbol="line-ns-open",
                                   line=dict(width=1.5)), name="Visited"),
        ],
        frames=frames,
        layout=go.Layout(
            xaxis=dict(range=[xr[0], xr[1]], title="q"),
            yaxis=dict(range=[-0.06, float(pd.max()) * 1.18], title="π(q)"),
            height=440,
            margin=dict(t=70, b=70, l=50, r=20),
            legend=dict(x=0.72, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=1.15, x=0.0, xanchor="left",
                buttons=[
                    dict(label="▶  Play", method="animate",
                         args=[None, dict(frame=dict(duration=420, redraw=True),
                                          fromcurrent=True, mode="immediate")]),
                    dict(label="⏸  Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
                ],
            )],
            sliders=[dict(
                active=0,
                currentvalue=dict(prefix="Step ", font=dict(size=11)),
                pad=dict(t=35),
                steps=[dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True),
                                                  mode="immediate")],
                            label=f.name, method="animate")
                       for f in frames],
            )],
        ),
    )
    return fig


@st.cache_data(show_spinner=False)
def build_leapfrog_anim(target_name, q0, p0, eps, L):
    """Plotly animation: leapfrog trajectory builds up step-by-step in phase space."""
    t = TARGETS[target_name]
    fn, xr = t["fn"], t["range"]

    _, _, traj = leapfrog(q0, p0, fn, eps, L)
    traj_eu   = euler_integration(q0, p0, fn, eps, L)

    qs = np.linspace(*xr, 100)
    ps = np.linspace(-4.5, 4.5, 100)
    U_grid = np.array([[-safe_log(fn, q) for q in qs] for _ in ps])
    _, P_grid = np.meshgrid(qs, ps)
    H_grid = U_grid + 0.5 * P_grid ** 2

    lf_q = [s[0] for s in traj];  lf_p = [s[1] for s in traj]
    eu_q = [s[0] for s in traj_eu]; eu_p = [s[1] for s in traj_eu]
    H0 = traj[0][2]

    frames = []
    for i in range(len(traj)):
        frames.append(go.Frame(
            name=str(i),
            traces=[3, 4],
            data=[
                go.Scatter(x=lf_q[:i+1], y=lf_p[:i+1], mode="lines+markers",
                           line=dict(color="#EF553B", width=2.5),
                           marker=dict(size=8, color="#EF553B")),
                go.Scatter(x=[lf_q[i]], y=[lf_p[i]], mode="markers",
                           marker=dict(size=18, color="crimson", symbol="star",
                                       line=dict(width=2, color="darkred"))),
            ],
        ))

    fig = go.Figure(
        data=[
            go.Contour(x=qs, y=ps, z=H_grid, colorscale="Blues",
                       contours=dict(start=max(0., H0 - 2.), end=H0 + 6., size=0.4,
                                     showlabels=False),
                       showscale=False, opacity=0.45, name="H contours"),
            go.Scatter(x=eu_q, y=eu_p, mode="lines+markers",
                       line=dict(color="#FFA15A", width=1.8, dash="dash"),
                       marker=dict(size=6, color="#FFA15A"), name="Euler (reference)"),
            go.Scatter(x=[lf_q[0]], y=[lf_p[0]], mode="markers",
                       marker=dict(size=16, color="limegreen", symbol="star"),
                       name="Start"),
            go.Scatter(x=[lf_q[0]], y=[lf_p[0]], mode="markers",
                       marker=dict(size=8, color="#EF553B"), name="Leapfrog"),
            go.Scatter(x=[lf_q[0]], y=[lf_p[0]], mode="markers",
                       marker=dict(size=18, color="crimson", symbol="star",
                                   line=dict(width=2, color="darkred")), name="Current"),
        ],
        frames=frames,
        layout=go.Layout(
            xaxis=dict(title="q  (position)"), yaxis=dict(title="p  (momentum)"),
            height=440, margin=dict(t=70, b=70),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=1.15, x=0.0, xanchor="left",
                buttons=[
                    dict(label="▶  Play", method="animate",
                         args=[None, dict(frame=dict(duration=500, redraw=True),
                                          fromcurrent=True)]),
                    dict(label="⏸  Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
                ],
            )],
            sliders=[dict(
                active=0,
                currentvalue=dict(prefix="Step ", font=dict(size=11)),
                pad=dict(t=35),
                steps=[dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True),
                                                  mode="immediate")],
                            label=f.name, method="animate")
                       for f in frames],
            )],
        ),
    )
    return fig


@st.cache_data(show_spinner=False)
def build_hmc_samples_anim(target_name, n_samples, eps, L, seed):
    """Plotly animation: HMC samples appear on the density one by one."""
    t = TARGETS[target_name]
    fn, xr = t["fn"], t["range"]

    rng = np.random.default_rng(int(seed))
    q = float(t["q0"])
    positions = [q]
    U = lambda q_: -safe_log(fn, q_)
    for _ in range(n_samples):
        p = float(rng.standard_normal())
        q_new, p_new, _ = leapfrog(q, p, fn, eps, L)
        dH = (U(q_new) + 0.5*p_new**2) - (U(q) + 0.5*p**2)
        if np.log(rng.random() + 1e-300) < -dH:
            q = q_new
        positions.append(q)

    xs = np.linspace(*xr, 300)
    lps = np.array([safe_log(fn, x) for x in xs])
    pd = np.exp(lps - lps.max())
    pd /= pd.sum() * (xr[1] - xr[0]) / 300

    def y_at(x): return float(np.interp(x, xs, pd))

    frames = []
    for i in range(1, len(positions)):
        trail = positions[:i]
        curr  = positions[i]
        frames.append(go.Frame(
            name=str(i),
            traces=[2, 3, 4],
            data=[
                go.Scatter(x=trail, y=[y_at(x) for x in trail], mode="markers",
                           marker=dict(size=7, color="#EF553B", opacity=0.3)),
                go.Scatter(x=[curr], y=[y_at(curr)], mode="markers",
                           marker=dict(size=22, color="#EF553B", symbol="star",
                                       line=dict(width=2, color="darkred"))),
                go.Scatter(x=positions[:i+1], y=[-0.025] * (i+1), mode="markers",
                           marker=dict(size=4, color="#EF553B", symbol="line-ns-open",
                                       line=dict(width=1.5))),
            ],
        ))

    fig = go.Figure(
        data=[
            go.Scatter(x=xs, y=pd, fill="tozeroy", mode="none",
                       fillcolor="rgba(239,85,59,0.08)", showlegend=False),
            go.Scatter(x=xs, y=pd, mode="lines",
                       line=dict(color="#00CC96", width=2.5), name="π(q)"),
            go.Scatter(x=[], y=[], mode="markers",
                       marker=dict(size=7, color="#EF553B", opacity=0.3), name="Past samples"),
            go.Scatter(x=[positions[0]], y=[y_at(positions[0])], mode="markers",
                       marker=dict(size=22, color="#EF553B", symbol="star",
                                   line=dict(width=2, color="darkred")), name="Current"),
            go.Scatter(x=[positions[0]], y=[-0.025], mode="markers",
                       marker=dict(size=4, color="#EF553B", symbol="line-ns-open",
                                   line=dict(width=1.5)), name="Visited"),
        ],
        frames=frames,
        layout=go.Layout(
            xaxis=dict(range=[xr[0], xr[1]], title="q"),
            yaxis=dict(range=[-0.06, float(pd.max()) * 1.18], title="π(q)"),
            height=440, margin=dict(t=70, b=70, l=50, r=20),
            legend=dict(x=0.72, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=1.15, x=0.0, xanchor="left",
                buttons=[
                    dict(label="▶  Play", method="animate",
                         args=[None, dict(frame=dict(duration=250, redraw=True),
                                          fromcurrent=True)]),
                    dict(label="⏸  Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
                ],
            )],
            sliders=[dict(
                active=0,
                currentvalue=dict(prefix="Sample ", font=dict(size=11)),
                pad=dict(t=35),
                steps=[dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True),
                                                  mode="immediate")],
                            label=f.name, method="animate")
                       for f in frames],
            )],
        ),
    )
    return fig

# ── Static plotting helpers ───────────────────────────────────────────────────

C_BLUE, C_RED, C_GREEN, C_ORANGE = "#636EFA", "#EF553B", "#00CC96", "#FFA15A"

def density_curve(log_target, x_range, n=600):
    xs = np.linspace(*x_range, n)
    lp = np.array([safe_log(log_target, x) for x in xs])
    p  = np.exp(lp - lp.max())
    p /= p.sum() * (x_range[1] - x_range[0]) / n
    return xs, p

def make_diagnostics(samples, log_target, x_range, color, title=""):
    acf_v = compute_acf(samples)
    xs, pd = density_curve(log_target, x_range)
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Trace", "Samples vs. target", "ACF"])
    fig.add_trace(go.Scatter(y=samples, mode="lines",
                             line=dict(color=color, width=0.6), showlegend=False), 1, 1)
    fig.add_trace(go.Histogram(x=samples, histnorm="probability density",
                               marker_color=color, opacity=0.55, nbinsx=60,
                               showlegend=False), 1, 2)
    fig.add_trace(go.Scatter(x=xs, y=pd, mode="lines",
                             line=dict(color=C_GREEN, width=2.5), showlegend=False), 1, 2)
    fig.add_trace(go.Bar(x=list(range(1, len(acf_v)+1)), y=acf_v,
                         marker_color=color, showlegend=False), 1, 3)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=3)
    fig.update_layout(title=title, height=320, margin=dict(t=60, b=20))
    return fig

def make_phase_space(log_target, x_range, traj_lf, traj_eu=None, title="Phase space"):
    qs = np.linspace(*x_range, 120)
    ps = np.linspace(-4.5, 4.5, 120)
    U_g = np.array([[-safe_log(log_target, q) for q in qs] for _ in ps])
    _, PG = np.meshgrid(qs, ps)
    H_g = U_g + 0.5 * PG**2
    lq, lp, lH = zip(*traj_lf)
    H0 = lH[0]
    fig = go.Figure()
    fig.add_trace(go.Contour(x=qs, y=ps, z=H_g, colorscale="Blues",
                             contours=dict(start=max(0., H0-2.), end=H0+6., size=0.4,
                                           showlabels=False),
                             showscale=False, opacity=0.45))
    if traj_eu:
        eq, ep, _ = zip(*traj_eu)
        fig.add_trace(go.Scatter(x=eq, y=ep, mode="lines+markers",
                                 line=dict(color=C_ORANGE, width=2, dash="dash"),
                                 marker=dict(size=5, color=C_ORANGE), name="Euler"))
    fig.add_trace(go.Scatter(x=lq, y=lp, mode="lines+markers",
                             line=dict(color=C_RED, width=2.5),
                             marker=dict(size=8, color=C_RED), name="Leapfrog"))
    fig.add_trace(go.Scatter(x=[lq[0]], y=[lp[0]], mode="markers",
                             marker=dict(size=15, color="limegreen", symbol="star"),
                             name="Start"))
    fig.add_trace(go.Scatter(x=[lq[-1]], y=[lp[-1]], mode="markers",
                             marker=dict(size=12, color="crimson", symbol="x-thin-open",
                                         line=dict(width=3)), name="End"))
    fig.update_layout(title=title, xaxis_title="q", yaxis_title="p",
                      height=420, margin=dict(t=50, b=30),
                      legend=dict(x=0.01, y=0.99))
    return fig

def make_hamiltonian_fig(traj_lf, traj_eu=None):
    _, _, lH = zip(*traj_lf)
    H0 = lH[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(lH))), y=list(lH), mode="lines+markers",
                             line=dict(color=C_RED, width=2.5), name="Leapfrog"))
    if traj_eu:
        _, _, eH = zip(*traj_eu)
        fig.add_trace(go.Scatter(x=list(range(len(eH))), y=list(eH), mode="lines+markers",
                                 line=dict(color=C_ORANGE, width=2, dash="dash"),
                                 name="Euler"))
    fig.add_hline(y=H0, line_dash="dot", line_color="gray",
                  annotation_text="H₀", annotation_position="bottom right")
    fig.update_layout(title="H(q, p) along trajectory — should stay ≈ flat",
                      xaxis_title="Step", yaxis_title="H(q, p)",
                      height=320, margin=dict(t=50, b=30))
    return fig

# ── Step navigation helper ────────────────────────────────────────────────────

def step_nav(key, labels):
    """Render step indicator + prev/next buttons. Returns current step index."""
    n = len(labels)
    if key not in st.session_state:
        st.session_state[key] = 0
    step = st.session_state[key]

    c_prev, c_dots, c_next = st.columns([1, 8, 1])
    with c_prev:
        if st.button("◀", key=f"__{key}_p", disabled=(step == 0), help="Previous"):
            st.session_state[key] -= 1
            st.rerun()
    with c_dots:
        filled   = "".join(f'<span style="color:#636EFA;font-size:1.3em">●</span>' if i == step
                           else f'<span style="color:#ccc;font-size:1.1em">○</span>'
                           for i in range(n))
        st.markdown(
            f'<div style="text-align:center;line-height:2em">{filled}'
            f'&ensp;<b style="color:#555;font-size:0.85em">{step+1}&thinsp;/&thinsp;{n}'
            f'&ensp;—&ensp;{labels[step]}</b></div>',
            unsafe_allow_html=True,
        )
    with c_next:
        lbl = "Next ▶" if step < n - 1 else "—"
        if st.button(lbl, key=f"__{key}_n", disabled=(step == n - 1),
                     type="primary" if step < n - 1 else "secondary"):
            st.session_state[key] += 1
            st.rerun()

    st.divider()
    return step

# ── Phenology helpers ──────────────────────────────────────────────────────────

def integrate_gdd_fast(T_arr: np.ndarray, T_base: float,
                       H_star: float, t0_int: int) -> np.ndarray:
    """
    Vectorised GDD integration.

    For each year, zero out heat on days < t0, cumsum the daily GDD, and
    return the first day where H_cum >= H_star (1-indexed DOY).
    Falls back to 365 for years that never reach the threshold.
    """
    t0_int = max(0, min(int(t0_int), T_arr.shape[1] - 1))
    gdd    = np.maximum(T_arr - T_base, 0.0)
    gdd[:, :t0_int] = 0.0
    H_cum  = np.cumsum(gdd, axis=1)
    crossed = H_cum >= H_star
    has_crossed = crossed.any(axis=1)
    return np.where(has_crossed,
                    np.argmax(crossed, axis=1).astype(float) + 1.0,
                    365.0)


@st.cache_data(show_spinner=False)
def generate_phenology_data():
    """Simulate 40 years of GDD-driven bloom observations."""
    np.random.seed(42)
    N_YEARS        = 40
    TRUE_T_BASE    = 3.0
    TRUE_H_STAR    = 220.0
    TRUE_T0        = 50
    TRUE_SIGMA     = 4.0
    years          = np.arange(1980, 1980 + N_YEARS)

    def _daily_temps(year, n_days=365, warming_rate=0.03):
        t        = np.arange(n_days)
        baseline = -8 + 18 * np.sin(np.pi * (t - 60) / 180)
        trend    = warming_rate * (year - 1980)
        noise    = np.random.normal(0, 3, n_days)
        return baseline + trend + noise

    T_daily  = np.array([_daily_temps(yr) for yr in years])
    doy_true = integrate_gdd_fast(T_daily, TRUE_T_BASE, TRUE_H_STAR, TRUE_T0)
    doy_obs  = np.round(doy_true + np.random.normal(0, TRUE_SIGMA, N_YEARS)).astype(float)

    true_params = dict(T_base=TRUE_T_BASE, H_star=TRUE_H_STAR,
                       t0=float(TRUE_T0),  sigma=TRUE_SIGMA)
    return T_daily, doy_obs, years, true_params


def make_log_prob_pheno(doy_obs_arr: np.ndarray, T_daily_arr: np.ndarray):
    """
    Returns a log_prob callable (state dict → float) that combines:
      - priors:  T_base ~ N(2,3),  H_star ~ HalfNormal(200),
                 t0 ~ Uniform(1,120),  sigma ~ HalfNormal(10)
      - likelihood: DOY ~ N(t*(theta), sigma)

    # ── To use real data ──────────────────────────────────────────────────
    # Bloom observations:
    #   RMBL Phenology Project: osf.io/jt4n5
    #   USA-NPN (Montana species): usanpn.org/data
    #   Species to try: Delphinium barbeyi or Mertensia ciliata
    #
    # Temperature forcing:
    #   NOAA GHCN daily: ncei.noaa.gov  (Crested Butte CO / Missoula MT)
    #   Daymet gridded 1-km daily: daymet.ornl.gov
    #
    # Expected shapes:
    #   doy_obs_arr : (n_years,)   float   — observed first-bloom DOY
    #   T_daily_arr : (n_years, 365) float  — daily mean temperatures (°C)
    # ─────────────────────────────────────────────────────────────────────
    """
    import scipy.stats as _sp2

    def log_prob(state: dict) -> float:
        T_base = state["T_base"]
        H_star = state["H_star"]
        t0_val = state["t0"]
        sigma  = state["sigma"]

        if sigma <= 0 or H_star <= 0:
            return -np.inf
        if t0_val < 1.0 or t0_val > 120.0:
            return -np.inf

        # Priors
        lp  = float(_sp2.norm.logpdf(T_base, 2.0, 3.0))
        lp += float(_sp2.halfnorm.logpdf(H_star, scale=200.0))
        lp += float(_sp2.uniform.logpdf(t0_val, 1.0, 119.0))
        lp += float(_sp2.halfnorm.logpdf(sigma, scale=10.0))

        # Likelihood
        doy_pred = integrate_gdd_fast(T_daily_arr, T_base, H_star, int(round(t0_val)))
        lp += float(np.sum(_sp2.norm.logpdf(doy_obs_arr, doy_pred, sigma)))
        return lp

    return log_prob


# ── Session state init ────────────────────────────────────────────────────────

for _k in ("ov_step", "mh_step", "hmc_step", "ppl_step", "mh_res", "hmc_res",
           "ppl_samples", "pheno_ppl_res", "pheno_pymc_res"):
    if _k not in st.session_state:
        st.session_state[_k] = None if _k.endswith(("res", "samples")) else 0

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_ov, tab_mh, tab_hmc, tab_cmp, tab_ppl, tab_pheno = st.tabs([
    "Overview", "Metropolis-Hastings", "HMC & Leapfrog", "Comparison",
    "🔧 Build a PPL", "🌸 Wildflower Phenology",
])

# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_ov:
    st.title("Markov Chain Monte Carlo — Interactive Explorer")

    OV_LABELS = [
        "Point estimates lie",
        "Bayesian updating",
        "The case for Bayes",
        "The intractability wall",
        "MCMC to the rescue",
    ]
    ov_step = step_nav("ov_step", OV_LABELS)

    # ── Step 0: Point estimates lie ────────────────────────────────────────────
    if ov_step == 0:
        st.markdown("## Your model doesn't know what it doesn't know")
        col_l, col_r = st.columns([1, 1])

        with col_l:
            st.markdown("""
You train a classifier. It hits **94 % accuracy** on the test set. You ship it.

Six months later it confidently mislabels a critical input and nobody knows why —
the model said **99.9 % confident**.

**The root problem: a single number cannot represent uncertainty.**

Standard ML training finds one answer — the *maximum likelihood estimate*:
""")
            st.latex(r"\hat{\theta} = \arg\max_\theta \;\log p(y \mid \theta)")
            st.markdown("""
That's a point in parameter space. It tells you which parameters fit the data *best*.
It says nothing about how many *other* parameter settings fit almost as well.

**Bayesian inference** asks a different question:

> Given the data I observed, what is the full *distribution* of plausible parameters?

The answer is the **posterior** — a probability distribution over θ that is *wide*
when data are scarce and *narrow* when data are abundant.
""")
            st.info("""
**The Bayesian promise**

| Situation | Point estimate says | Posterior says |
|---|---|---|
| 3 training examples | θ̂ = 0.71 (confident) | Very uncertain — posterior is wide |
| 3 000 training examples | θ̂ = 0.71 (confident) | Confident — posterior is narrow |

The uncertainty is *quantified*, not discarded.
""")

        with col_r:
            st.markdown("### Watch uncertainty shrink as data arrive")
            n_flips_0 = st.slider("Number of coin flips observed (N)",
                                  0, 120, 6, key="ov_nflips")
            np.random.seed(42)
            _coin_data = (np.random.rand(120) < 0.65).astype(int)
            n_heads_0  = int(_coin_data[:n_flips_0].sum())
            n_tails_0  = n_flips_0 - n_heads_0

            theta_r = np.linspace(0.001, 0.999, 400)
            # Prior: Beta(2,2) — mild preference for a fair coin
            prior_y   = stats.beta.pdf(theta_r, 2, 2)
            post_y    = stats.beta.pdf(theta_r, 2 + n_heads_0, 2 + n_tails_0)

            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(
                x=theta_r, y=prior_y, mode="lines",
                line=dict(color=C_ORANGE, width=2, dash="dash"),
                name="Prior  Beta(2, 2)",
                fill="tozeroy", fillcolor="rgba(255,161,90,0.10)",
            ))
            fig_beta.add_trace(go.Scatter(
                x=theta_r, y=post_y, mode="lines",
                line=dict(color=C_BLUE, width=3),
                name=f"Posterior  Beta({2+n_heads_0}, {2+n_tails_0})",
                fill="tozeroy", fillcolor="rgba(99,110,250,0.15)",
            ))
            fig_beta.add_vline(x=0.65, line_color=C_RED, line_dash="dot", line_width=2,
                               annotation_text="true p = 0.65",
                               annotation_position="top right")
            if n_flips_0 > 0:
                mle_p0 = n_heads_0 / n_flips_0
                fig_beta.add_vline(x=mle_p0, line_color=C_GREEN, line_dash="dot",
                                   line_width=2,
                                   annotation_text=f"MLE = {mle_p0:.2f}",
                                   annotation_position="top left")

            ci_lo0, ci_hi0 = stats.beta.ppf(
                [0.025, 0.975], 2 + n_heads_0, 2 + n_tails_0)
            post_mean0 = (2 + n_heads_0) / (4 + n_flips_0)

            fig_beta.update_layout(
                title=f"N = {n_flips_0}  ({n_heads_0} heads, {n_tails_0} tails)",
                xaxis_title="θ  =  P(heads)",
                yaxis_title="Probability density",
                height=310, margin=dict(t=50, b=30),
                legend=dict(x=0.02, y=0.95),
            )
            st.plotly_chart(fig_beta, use_container_width=True)

            st.markdown(f"""
**Posterior summary**
Posterior mean: **{post_mean0:.3f}** &nbsp;·&nbsp;
95 % credible interval: **[{ci_lo0:.3f}, {ci_hi0:.3f}]**

*Slide to N = 3 then N = 100.*
At N = 3 the 95 % CI is very wide — Bayes knows it knows very little.
At N = 100 the CI has collapsed around the truth.
The MLE is always a single dot; it never conveys this.
""")
            st.info("Click **Next ▶** to see exactly how the posterior is computed.")

    # ── Step 1: Bayesian updating ──────────────────────────────────────────────
    elif ov_step == 1:
        st.markdown("## Bayesian inference: belief as a first-class value")

        col_l, col_r = st.columns([1, 1])

        with col_l:
            st.markdown("### Bayes' theorem")
            st.latex(r"""
\underbrace{p(\theta \mid y)}_{\text{posterior}}
= \frac{
    \underbrace{p(y \mid \theta)}_{\text{likelihood}}
    \;\cdot\;
    \underbrace{p(\theta)}_{\text{prior}}
}{
    \underbrace{p(y)}_{Z \;=\; \text{normalising constant}}
}
""")
            st.markdown("""
| Term | What it encodes |
|------|-----------------|
| **Prior** p(θ) | Your belief *before* seeing any data |
| **Likelihood** p(y \| θ) | How probable the observed data is for each θ |
| **Posterior** p(θ \| y) | Updated belief *after* seeing the data |
| **Evidence Z** | A constant that makes the posterior integrate to 1 |

Because Z does not depend on θ, we almost always write:
""")
            st.latex(r"p(\theta \mid y) \;\propto\; p(y \mid \theta) \cdot p(\theta)")
            st.markdown("""
**Bayesian updating is sequential:**
today's posterior becomes tomorrow's prior when new data arrive —
no retraining from scratch, just another application of Bayes' rule.

**Credible intervals are literal probabilities:**
a 95 % Bayesian credible interval means there is a 95 % probability
that θ lies inside it, *given the data you observed*.
A 95 % frequentist confidence interval means no such thing.
""")

        with col_r:
            st.markdown("### A CS example you already know: the spam filter")
            st.markdown(r"""
The Naive Bayes classifier — probably the first Bayesian model you encountered —
is just Bayes' theorem applied to text classification:

$$P(\text{spam} \mid \text{email}) \;\propto\; P(\text{email} \mid \text{spam}) \cdot P(\text{spam})$$

- **Prior** P(spam): base rate of spam in your inbox
- **Likelihood** P(email \| spam): how common this word sequence is in spam
- **Posterior** P(spam \| email): the score your filter actually thresholds on

It's called *Naive* because it assumes words are conditionally independent given the
class — a simplification that makes Z tractable by factoring into per-word terms.

**Real Bayesian models drop that assumption.** When the model is a neural net,
a logistic regression with interactions, or a hierarchical model, Z becomes
an intractable high-dimensional integral. That is where MCMC enters.

---
""")
            st.markdown("### Prior sensitivity — 10 coin flips (7 heads, 3 tails)")
            theta_r1 = np.linspace(0.001, 0.999, 400)
            prior_specs = [
                ("Uninformative  Beta(1, 1)", 1,  1,  C_ORANGE),
                ("Weak fair-coin  Beta(2, 2)", 2,  2,  C_GREEN),
                ("Strong fair-coin  Beta(10,10)", 10, 10, C_RED),
            ]
            fig_sens = go.Figure()
            for label, a, b, col in prior_specs:
                post1 = stats.beta.pdf(theta_r1, a + 7, b + 3)
                fig_sens.add_trace(go.Scatter(
                    x=theta_r1, y=post1, mode="lines",
                    line=dict(color=col, width=2.5), name=label,
                ))
            fig_sens.add_vline(x=0.70, line_color="gray", line_dash="dot",
                               annotation_text="MLE = 0.70",
                               annotation_position="top left")
            fig_sens.update_layout(
                title="Same data, different priors → different posteriors",
                xaxis_title="θ = P(heads)", yaxis_title="Density",
                height=250, margin=dict(t=50, b=30),
                legend=dict(x=0.01, y=0.97, font=dict(size=10)),
            )
            st.plotly_chart(fig_sens, use_container_width=True)
            st.caption(
                "With only 10 flips a strong prior still dominates. "
                "With 100+ flips all three posteriors converge near the truth — "
                "data eventually overwhelms any reasonable prior."
            )

    # ── Step 2: The case for Bayes ─────────────────────────────────────────────
    elif ov_step == 2:
        st.markdown("## The case for Bayesian inference")

        col_l, col_r = st.columns([1, 1])

        with col_l:
            st.markdown("""
### What you lose with a point estimate

| | MLE / MAP | Full Bayesian |
|---|---|---|
| **Answer** | One parameter vector θ̂ | Distribution p(θ \| y) |
| **Uncertainty** | ✗ Not quantified | ✓ Built in |
| **Small data** | Overconfident, often wrong | Wide posterior = honest |
| **Predictions** | Single curve | Ensemble of curves |
| **Downstream decisions** | No uncertainty propagation | Uncertainty flows through |

**The small-data problem is real:**
with 6 data points and 2 parameters, MLE gives one confident answer.
But many parameter combinations fit the data almost equally well —
and that ambiguity matters for decisions.
""")
            st.info("""
**Practical example:**
You're building a recommendation model for a user with **4 purchase history items**.

- **MLE says:** "outdoor gear enthusiast, confidence 0.91"
- **Bayes says:** "probably outdoor gear, but posterior is very wide —
  show a hedge of diverse items until more data arrive"

The Bayesian recommendation system *knows it doesn't know* and acts accordingly.
""")

        with col_r:
            st.markdown("### Posterior vs MAP — 6 data points")
            st.markdown("""
Below: 6 students, hours studied vs pass/fail.
Each **blue curve** is one draw from the posterior — a plausible logistic model.
The **red curve** is the MAP estimate (single best answer).

With only 6 points, the posterior is a *family* of curves, not one line.
""")
            x_tiny = np.array([1.0, 2.0, 4.0, 5.0, 7.5, 9.0])
            y_tiny = np.array([0,   0,   0,   1,   1,   1  ])

            def _lp_tiny(b0, b1):
                z  = np.clip(b0 + b1 * x_tiny, -30, 30)
                ll = float(np.sum(y_tiny * (-np.log1p(np.exp(-z)))
                                  + (1 - y_tiny) * (-np.log1p(np.exp(z)))))
                lp = -0.5 * (b0**2 + b1**2) / 4.0
                return ll + lp

            b0g2 = np.linspace(-6, 4, 120)
            b1g2 = np.linspace(-0.3, 2.5, 120)
            logp2 = np.array([[_lp_tiny(b0, b1) for b0 in b0g2]
                               for b1 in b1g2])
            logp2 -= logp2.max()
            prob2  = np.exp(logp2)
            prob2 /= prob2.sum()

            rng2     = np.random.default_rng(77)
            flat_idx = rng2.choice(prob2.size, size=80, p=prob2.ravel())
            post_b0  = b0g2[flat_idx % len(b0g2)]
            post_b1  = b1g2[flat_idx // len(b0g2)]

            x_cv2 = np.linspace(0, 10, 200)
            fig_lr2 = go.Figure()

            for pb0, pb1 in zip(post_b0, post_b1):
                pc = 1.0 / (1.0 + np.exp(-(pb0 + pb1 * x_cv2)))
                fig_lr2.add_trace(go.Scatter(
                    x=x_cv2, y=pc, mode="lines",
                    line=dict(color="rgba(99,110,250,0.07)", width=1.5),
                    showlegend=False,
                ))

            best2   = np.unravel_index(logp2.argmax(), logp2.shape)
            b0_map2 = b0g2[best2[1]]; b1_map2 = b1g2[best2[0]]
            p_map2  = 1.0 / (1.0 + np.exp(-(b0_map2 + b1_map2 * x_cv2)))
            fig_lr2.add_trace(go.Scatter(
                x=x_cv2, y=p_map2, mode="lines",
                line=dict(color=C_RED, width=3), name="MAP (single estimate)",
            ))
            fig_lr2.add_trace(go.Scatter(
                x=x_tiny[y_tiny == 0], y=y_tiny[y_tiny == 0].astype(float),
                mode="markers", marker=dict(size=14, color=C_RED, symbol="x"),
                name="Fail",
            ))
            fig_lr2.add_trace(go.Scatter(
                x=x_tiny[y_tiny == 1], y=y_tiny[y_tiny == 1].astype(float),
                mode="markers", marker=dict(size=14, color=C_GREEN),
                name="Pass",
            ))
            fig_lr2.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color="rgba(99,110,250,0.5)", width=2),
                name="Posterior samples",
            ))
            fig_lr2.update_layout(
                title="Posterior samples vs MAP — 6 training points",
                xaxis_title="Hours studied",
                yaxis=dict(title="P(pass)", range=[-0.05, 1.05]),
                height=330, margin=dict(t=50, b=30),
                legend=dict(x=0.02, y=0.95),
            )
            st.plotly_chart(fig_lr2, use_container_width=True)
            st.caption(
                "MAP picks the most likely single curve and discards all ambiguity. "
                "The posterior retains every plausible curve — weighted by how well "
                "it fits the data and how reasonable the prior considers it."
            )

    # ── Step 3: The intractability wall ────────────────────────────────────────
    elif ov_step == 3:
        st.markdown("## The intractability wall")

        col_l, col_r = st.columns([1, 1])

        with col_l:
            st.markdown("""
### Why not just compute the posterior directly?

The posterior requires the normalising constant Z:
""")
            st.latex(r"p(\theta \mid y) = \frac{p(y \mid \theta)\,p(\theta)}{Z}, \qquad Z = \int p(y \mid \theta)\,p(\theta)\;d\theta")
            st.markdown("""
For almost all real models Z has **no closed form.**

The only alternative is numerical integration — but this fails catastrophically
as the number of parameters grows.

**Grid integration cost: G^D evaluations**
""")
            grid_data = {
                "Parameters D": [1, 2, 5, 10, 20, 50],
                "Grid pts / dim": [1000, 100, 100, 100, 10, 10],
                "Evaluations needed": ["10³", "10⁴", "10¹⁰", "10²⁰", "10²⁰", "10⁵⁰"],
                "Time @ 10⁹/s": ["instant", "instant", "10 s", "3×10³ yr", "3×10³ yr", "3×10³³ yr"],
            }
            st.dataframe(pd.DataFrame(grid_data), hide_index=True, use_container_width=True)
            st.error(
                "**The grid approach is dead on arrival for any real model.** "
                "Real Bayesian models routinely have 10–10 000 parameters."
            )
            st.markdown("""
**Variational inference** (VI) is one alternative — it approximates the posterior
with a simpler family of distributions and optimises.  Fast, but biased.

**MCMC** is exact in the limit and requires only *pointwise* evaluations of
the unnormalised posterior — no Z needed at all.
""")

        with col_r:
            st.markdown("### Curse of dimensionality — all the mass is in a shell")

            from scipy.special import gamma as _gamma_fn
            dims_cod = np.arange(1, 35)
            # Volume of d-ball as fraction of bounding d-cube
            ball_frac = (np.pi ** (dims_cod / 2.0)
                         / (_gamma_fn(dims_cod / 2.0 + 1) * 2.0 ** dims_cod))
            ball_frac = np.clip(ball_frac, 1e-300, 1.0)

            fig_cod2 = go.Figure()
            fig_cod2.add_trace(go.Scatter(
                x=dims_cod, y=ball_frac * 100.0,
                mode="lines+markers",
                line=dict(color=C_RED, width=2.5),
                marker=dict(size=5),
                name="% of cube volume in inscribed ball",
            ))
            fig_cod2.update_layout(
                title="In high dimensions, the cube is almost all corners",
                xaxis_title="Number of dimensions D",
                yaxis_title="% of cube volume in inscribed ball",
                yaxis=dict(type="log"),
                height=240, margin=dict(t=50, b=30),
            )
            st.plotly_chart(fig_cod2, use_container_width=True)

            st.markdown("""
In 10D, less than 0.25 % of the bounding box has any posterior mass inside it.
A uniform grid wastes 99.75 % of its evaluations.
The posterior mass lives on a thin **typical set** — a shell that grid methods
cannot find efficiently but MCMC navigates naturally.

---

### The key insight: Z cancels in MCMC

In the Metropolis acceptance step we take the *ratio* of posteriors:
""")
            st.latex(r"""
\alpha
= \min\!\left(1,\;\frac{p(\theta^* \mid y)}{p(\theta \mid y)}\right)
= \min\!\left(1,\;
  \frac{p(y \mid \theta^*)\,p(\theta^*)}{p(y \mid \theta)\,p(\theta)}
  \cdot\cancel{\frac{Z}{Z}}
\right)
""")
            st.success(
                "**Z cancels exactly.**  "
                "MCMC only needs the unnormalised product p(y|θ)·p(θ) — "
                "evaluated at two points.  No integrals, no Z."
            )

    # ── Step 4: MCMC to the rescue ─────────────────────────────────────────────
    elif ov_step == 4:
        st.markdown("## MCMC: exploring the posterior without Z")

        st.markdown("""
An MCMC algorithm constructs a Markov chain whose **stationary distribution is
exactly the posterior** p(θ|y).  Simulate the chain long enough and the histogram
of visited states converges to the posterior — *no Z required*.

**Only requirement:** evaluate the unnormalised posterior p̃(θ|y) = p(y|θ)·p(θ) pointwise.
""")

        col_l, col_r = st.columns([1, 1])

        with col_l:
            st.markdown("""#### Metropolis-Hastings (Metropolis et al., 1953)

Invented for nuclear-weapons simulations at Los Alamos.
From the current position θ:

1. **Propose** θ* ~ N(θ, σ²) &nbsp; *(random Gaussian step)*
2. Compute **acceptance ratio** α = min(1, p̃(θ*) / p̃(θ))
3. Accept θ* with probability α; otherwise stay at θ

The chain drifts toward high-probability regions because uphill moves always
accept; downhill moves accept with a probability proportional to the density
drop, so the chain never gets permanently trapped.

**Drawback:** moves like a random walk.  To travel distance d the chain needs
O(d²) steps — slow in high dimensions.
""")
            st.info("→ **Metropolis-Hastings tab** — step-by-step walkthrough with live visualisations")

        with col_r:
            st.markdown("""#### Hamiltonian Monte Carlo (Duane et al., 1987)

Invented for lattice QCD simulations.  Add an auxiliary **momentum** p ~ N(0, I)
and define the Hamiltonian:
""")
            st.latex(r"H(\theta, p) = \underbrace{-\log \tilde{p}(\theta \mid y)}_{U(\theta)\ \text{potential}} + \underbrace{\tfrac{1}{2}p^\top p}_{K(p)\ \text{kinetic}}")
            st.markdown("""
Run **leapfrog integration** (a symplectic ODE solver) for L steps at step size ε.
The leapfrog trajectory follows constant-H surfaces, so the proposal lands *far*
from the start without the random-walk diffusion of MH.

Accept/reject on ΔH to correct for discretisation error.

**Requires:** ∇_θ log p̃(θ|y) — the gradient, computed by automatic differentiation
in PyMC / Stan / NumPyro.
""")
            st.info("→ **HMC & Leapfrog tab** — leapfrog integrator + phase portrait visualisations")

        st.divider()
        st.markdown("### At a glance")
        st.markdown("""
| | **Metropolis-Hastings** | **Hamiltonian Monte Carlo** |
|---|---|---|
| **Year / origin** | 1953 — nuclear physics | 1987 — lattice QCD |
| **Proposal** | Random Gaussian jump | Gradient-guided leapfrog trajectory |
| **Needs** | log p̃(θ) only | log p̃(θ) + **gradient** ∇ log p̃(θ) |
| **Mixing speed** | Slow — random walk O(d²) | Fast — O(d^{5/4}) steps |
| **Acceptance rate** | Target ~23–50 % | Target ~60–90 % |
| **Tuning knobs** | Proposal width σ | Step size ε, leapfrog steps L |
| **Used in** | Teaching, non-differentiable models | PyMC, Stan, NumPyro, Pyro |

Use the tabs above to step through each algorithm interactively,
then compare them head-to-head in the **Comparison** tab.
""")
        st.info("""
**Reading MCMC diagnostics** — used throughout the rest of this app

| Plot | What "good" looks like |
|---|---|
| **Trace plot** | Noisy, no trend, no long flat stretches |
| **Histogram** | Smooth, matches the target density |
| **ACF** | Drops quickly toward zero |
| **Acceptance rate** | MH: 20–50 % · HMC: 60–90 % |
""")

# ─────────────────────────────────────────────────────────────────────────────
# METROPOLIS-HASTINGS TAB
# ─────────────────────────────────────────────────────────────────────────────
with tab_mh:
    st.header("Metropolis-Hastings")

    # Target selector persists across all steps
    tc, dc = st.columns([2, 4])
    with tc:
        mh_tgt = st.selectbox("Target distribution", list(TARGETS.keys()), key="mh_tgt")
    with dc:
        st.caption(f"*{TARGETS[mh_tgt]['desc']}*")

    MH_LABELS = ["The target", "Proposals", "Accept / Reject", "The chain walks", "Explore"]
    step = step_nav("mh_step", MH_LABELS)

    fn_mh = TARGETS[mh_tgt]["fn"]
    xr_mh = TARGETS[mh_tgt]["range"]

    # ── Step 0: The Target ─────────────────────────────────────────────────────
    if step == 0:
        L, R = st.columns([1, 2])
        with L:
            st.markdown("## The target distribution")
            st.markdown("""
We want samples from π(q) — perhaps a Bayesian posterior or a physical distribution.

We **cannot** sample directly, but we **can** evaluate the unnormalized log-density at any point:
""")
            st.latex(r"\log\tilde{\pi}(q) = \log\pi(q) + \text{const}")
            st.markdown("""
MCMC only needs these pointwise evaluations — the normalizing constant never matters.

The strategy: construct a **Markov chain** that wanders through q-space and, in the long run, visits each region in proportion to π(q).
""")
            st.info("Click **Next ▶** to see how we move the chain.")
        with R:
            xs_d, pd_d = density_curve(fn_mh, xr_mh)
            fig0 = go.Figure()
            fig0.add_trace(go.Scatter(x=xs_d, y=pd_d, fill="tozeroy", mode="none",
                                      fillcolor="rgba(0,204,150,0.15)", showlegend=False))
            fig0.add_trace(go.Scatter(x=xs_d, y=pd_d, mode="lines",
                                      line=dict(color=C_GREEN, width=3), name="π(q)"))
            fig0.update_layout(title="Target π(q)", xaxis_title="q", yaxis_title="π(q)",
                               height=380, margin=dict(t=60, b=20))
            st.plotly_chart(fig0, use_container_width=True)

    # ── Step 1: Proposals ─────────────────────────────────────────────────────
    elif step == 1:
        L, R = st.columns([1, 2])
        with L:
            st.markdown("## Making a proposal")
            st.markdown("From the current position **q**, we draw a candidate:")
            st.latex(r"q^* \;\sim\; \mathcal{N}(q,\;\sigma^2)")
            st.markdown("""
σ is the **proposal width** (step size). It controls how far we jump on average.

The proposal is **symmetric** — proposing q\* from q is just as likely as proposing q from q\*. This symmetry cancels the Hastings correction, leaving only the target ratio in the acceptance step.
""")
            st.markdown("---")
            demo_sig = st.slider("Try different σ values", 0.1, 4.0, 1.0, 0.1,
                                 key="mh_prop_demo")
            st.caption("Watch how the orange proposal cloud widens or narrows.")
        with R:
            xs_d, pd_d = density_curve(fn_mh, xr_mh)
            q_demo = float(np.percentile(xs_d, 25))  # somewhere off-center
            xs_prop = np.linspace(*xr_mh, 300)
            prop_y  = stats.norm.pdf(xs_prop, q_demo, demo_sig)
            prop_scaled = prop_y * float(pd_d.max()) / prop_y.max() * 0.55

            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=xs_d, y=pd_d, fill="tozeroy", mode="none",
                                      fillcolor="rgba(0,204,150,0.12)", showlegend=False))
            fig1.add_trace(go.Scatter(x=xs_d, y=pd_d, mode="lines",
                                      line=dict(color=C_GREEN, width=2.5), name="π(q)"))
            fig1.add_trace(go.Scatter(x=xs_prop, y=prop_scaled, mode="lines",
                                      fill="tozeroy", fillcolor="rgba(255,161,90,0.18)",
                                      line=dict(color=C_ORANGE, width=2, dash="dash"),
                                      name=f"N(q, σ²)  σ={demo_sig}"))
            yc = float(np.interp(q_demo, xs_d, pd_d))
            fig1.add_trace(go.Scatter(x=[q_demo], y=[yc], mode="markers",
                                      marker=dict(size=20, color=C_BLUE,
                                                  line=dict(width=2.5, color="#2929cc")),
                                      name="Current q"))
            fig1.add_annotation(x=q_demo + demo_sig, y=yc * 0.35,
                                 ax=q_demo, ay=yc * 0.35,
                                 xref="x", yref="y", axref="x", ayref="y",
                                 arrowhead=2, arrowsize=1.5, arrowwidth=2,
                                 arrowcolor=C_ORANGE)
            fig1.add_annotation(x=q_demo - demo_sig, y=yc * 0.35,
                                 ax=q_demo, ay=yc * 0.35,
                                 xref="x", yref="y", axref="x", ayref="y",
                                 arrowhead=2, arrowsize=1.5, arrowwidth=2,
                                 arrowcolor=C_ORANGE)
            fig1.add_annotation(text=f"±σ = ±{demo_sig:.1f}", x=q_demo, y=yc * 0.28,
                                 showarrow=False, font=dict(size=11, color=C_ORANGE))
            fig1.update_layout(title="Proposal distribution centered at current q",
                               xaxis_title="q", height=380, margin=dict(t=60, b=20))
            st.plotly_chart(fig1, use_container_width=True)

    # ── Step 2: Accept / Reject ────────────────────────────────────────────────
    elif step == 2:
        L, R = st.columns([1, 2])
        with L:
            st.markdown("## Accept or reject?")
            st.markdown("Compute the **log acceptance ratio**:")
            st.latex(r"\log\alpha = \log\pi(q^*) - \log\pi(q)")
            st.markdown("Draw u ~ Uniform(0,1). Accept if log u < log α.")
            st.markdown("""
**Two cases:**

🟢 **Uphill move** (q\* has higher density)
→ log α ≥ 0 → **always accept**, chain moves to q\*

🟡 **Downhill move** (q\* has lower density)
→ log α < 0 → accept **with probability α = π(q\*)/π(q)**

The chain spends more time in high-density regions *automatically*, because uphill moves are always accepted and downhill moves are only sometimes accepted.
""")
        with R:
            xs_d, pd_d = density_curve(fn_mh, xr_mh)
            # Pick illustrative examples
            q_up_c = float(np.percentile(xs_d, 20))
            q_up_p = float(np.percentile(xs_d, 45))
            q_dn_c = float(np.percentile(xs_d, 55))
            q_dn_p = float(np.percentile(xs_d, 85))

            y_uc = float(np.interp(q_up_c, xs_d, pd_d))
            y_up = float(np.interp(q_up_p, xs_d, pd_d))
            y_dc = float(np.interp(q_dn_c, xs_d, pd_d))
            y_dp = float(np.interp(q_dn_p, xs_d, pd_d))

            a_up = min(1., np.exp(safe_log(fn_mh, q_up_p) - safe_log(fn_mh, q_up_c)))
            a_dn = min(1., np.exp(safe_log(fn_mh, q_dn_p) - safe_log(fn_mh, q_dn_c)))

            fig2 = make_subplots(
                rows=1, cols=2,
                subplot_titles=[
                    f"Uphill: α = {a_up:.2f} → always accept ✓",
                    f"Downhill: α = {a_dn:.2f} → accept {a_dn:.0%} of the time",
                ],
            )
            for ci, (qc, qp, yc, yp, alpha) in enumerate([
                (q_up_c, q_up_p, y_uc, y_up, a_up),
                (q_dn_c, q_dn_p, y_dc, y_dp, a_dn),
            ], start=1):
                fig2.add_trace(go.Scatter(x=xs_d, y=pd_d, fill="tozeroy", mode="none",
                                          fillcolor="rgba(0,204,150,0.10)",
                                          showlegend=False), 1, ci)
                fig2.add_trace(go.Scatter(x=xs_d, y=pd_d, mode="lines",
                                          line=dict(color=C_GREEN, width=2),
                                          showlegend=False), 1, ci)
                for qx, yx, col, nm in [
                    (qc, yc, C_BLUE,   "Current q"),
                    (qp, yp, C_GREEN if alpha >= 1 else C_ORANGE, "Proposal q*"),
                ]:
                    fig2.add_shape(type="line", x0=qx, x1=qx, y0=0, y1=yx,
                                   line=dict(color=col, width=1.5, dash="dot"),
                                   row=1, col=ci)
                    fig2.add_trace(go.Scatter(x=[qx], y=[yx], mode="markers",
                                              marker=dict(size=16, color=col),
                                              name=nm, showlegend=(ci == 1)), 1, ci)
                # Arrow from current to proposal
                mid_y = (yc + yp) * 0.3
                fig2.add_annotation(x=qp, y=mid_y, ax=qc, ay=mid_y,
                                    xref=f"x{ci}", yref=f"y{ci}",
                                    axref=f"x{ci}", ayref=f"y{ci}",
                                    arrowhead=2, arrowsize=1.5, arrowwidth=2,
                                    arrowcolor="gray", showarrow=True)
            fig2.update_layout(height=400, margin=dict(t=80, b=20))
            st.plotly_chart(fig2, use_container_width=True)

    # ── Step 3: The Chain Walks ────────────────────────────────────────────────
    elif step == 3:
        L, R = st.columns([1, 2])
        with L:
            st.markdown("## The chain walks")
            st.markdown("""
Repeat propose → accept/reject over and over. The chain **wanders** through q-space, pulled toward high-density regions.

**Legend**
- 🔵 Blue ball = current position
- 🟠 Orange ball = proposal for this step
- 🟢 Green = accepted (ball jumps to proposal)
- 🔴 Red = rejected (ball stays put)
- Ticks at the bottom accumulate all visited positions
""")
            st.markdown("---")
            a_std  = st.slider("Proposal σ", 0.1, 4.0, 1.0, 0.1, key="mh_a_std")
            a_n    = st.slider("Steps to animate", 20, 80, 45, 5, key="mh_a_n")
            a_seed = st.number_input("Seed", 0, 999, 42, key="mh_a_seed")
            st.markdown("*Press ▶ Play in the chart, or drag the slider.*")
        with R:
            fig3 = build_metro_anim(mh_tgt, int(a_n), a_std, int(a_seed))
            st.plotly_chart(fig3, use_container_width=True)

    # ── Step 4: Explore ────────────────────────────────────────────────────────
    elif step == 4:
        st.markdown("## Explore — run the full sampler")
        L, R = st.columns([1, 3])
        with L:
            e_std  = st.slider("Proposal σ", 0.05, 5.0, 1.0, 0.05, key="mh_e_std")
            e_n    = st.slider("Samples", 500, 10_000, 3_000, 500, key="mh_e_n")
            e_burn = st.slider("Burn-in", 100, 2_000, 500, 100, key="mh_e_burn")
            e_seed = st.number_input("Seed", 0, 9_999, 42, key="mh_e_seed")
            if st.button("Run Metropolis", type="primary", key="run_mh_e"):
                with st.spinner("Running…"):
                    s, r = _cached_metro(mh_tgt, e_n, e_burn, e_std, int(e_seed))
                st.session_state["mh_res"] = dict(s=s, r=r, tgt=mh_tgt, std=e_std)
            st.markdown("---")
            st.markdown("""
**What to look for**

| σ | Acceptance | ACF |
|---|---|---|
| Too small | ~100% | Slow decay |
| ~Optimal | 20–50% | Fast decay |
| Too large | ~0% | Slow decay |
            """)
        with R:
            res = st.session_state["mh_res"]
            if res:
                c1, c2 = st.columns(2)
                c1.metric("Acceptance rate", f"{res['r']:.1%}")
                c2.metric("Proposal σ", f"{res['std']:.2f}")
                st.plotly_chart(
                    make_diagnostics(res["s"], TARGETS[res["tgt"]]["fn"],
                                     TARGETS[res["tgt"]]["range"],
                                     C_BLUE, f"Metropolis — {res['tgt']}"),
                    use_container_width=True,
                )
            else:
                st.info("Configure parameters and click **Run Metropolis**.")

# ─────────────────────────────────────────────────────────────────────────────
# HMC & LEAPFROG TAB
# ─────────────────────────────────────────────────────────────────────────────
with tab_hmc:
    st.header("Hamiltonian Monte Carlo & the Leapfrog Integrator")

    hc, dc = st.columns([2, 4])
    with hc:
        hmc_tgt = st.selectbox("Target distribution", list(TARGETS.keys()), key="hmc_tgt")
    with dc:
        st.caption(f"*{TARGETS[hmc_tgt]['desc']}*")

    HMC_LABELS = ["Random walks are slow", "Potential energy",
                  "Hamiltonian dynamics", "The leapfrog integrator",
                  "Euler vs. leapfrog", "HMC in action"]
    hstep = step_nav("hmc_step", HMC_LABELS)

    fn_hmc = TARGETS[hmc_tgt]["fn"]
    xr_hmc = TARGETS[hmc_tgt]["range"]

    # ── HMC Step 0: Random walks are slow ─────────────────────────────────────
    if hstep == 0:
        L, R = st.columns([1, 2])
        with L:
            st.markdown("## The problem with random walks")
            st.markdown("""
Metropolis explores via a **random walk**: each step is drawn from a Gaussian centred at the current position. This means:

- To travel distance d, it takes O(d²) steps
- Consecutive samples are **highly correlated**
- The ACF decays slowly → we need many samples to get the same effective information

The plot on the right shows Metropolis on the selected target with a moderate proposal width.
Notice the trace **creeping** rather than jumping, and the ACF staying elevated for many lags.
""")
            st.info("HMC solves this by using the **gradient** of log π to make long, directed proposals.")
        with R:
            with st.spinner("Computing Metropolis reference…"):
                s_slow, r_slow = _cached_metro(hmc_tgt, 2_000, 500, 0.5, 42)
            acf_slow = compute_acf(s_slow)
            xs_d, pd_d = density_curve(fn_hmc, xr_hmc)

            fig_s = make_subplots(rows=1, cols=2,
                                  subplot_titles=["Trace (random walk)", "Autocorrelation"])
            fig_s.add_trace(go.Scatter(y=s_slow, mode="lines",
                                       line=dict(color=C_BLUE, width=0.7),
                                       showlegend=False), 1, 1)
            fig_s.add_trace(go.Bar(x=list(range(1, len(acf_slow)+1)), y=acf_slow,
                                   marker_color=C_BLUE, showlegend=False), 1, 2)
            fig_s.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
            fig_s.add_annotation(text=f"ACF lag-1 = {acf_slow[0]:.3f}",
                                  xref="x2 domain", yref="y2 domain",
                                  x=0.95, y=0.95, showarrow=False,
                                  font=dict(size=13, color=C_BLUE),
                                  xanchor="right")
            fig_s.update_layout(height=380, margin=dict(t=60, b=20))
            st.plotly_chart(fig_s, use_container_width=True)

    # ── HMC Step 1: Potential energy landscape ─────────────────────────────────
    elif hstep == 1:
        L, R = st.columns([1, 2])
        with L:
            st.markdown("## The potential energy landscape")
            st.markdown("""
Define the **potential energy**:
""")
            st.latex(r"U(q) = -\log\pi(q)")
            st.markdown("""
This *inverts* the density — high probability regions become **valleys** (low energy), low probability regions become **hills** (high energy).

Imagine a ball rolling on this landscape. It naturally settles in the valleys and has to overcome energy barriers to cross between them.

HMC uses this physical intuition directly: it gives the ball **momentum** and lets it roll, exploring the landscape efficiently.
""")
        with R:
            xs_d, pd_d = density_curve(fn_hmc, xr_hmc)
            U_vals = np.array([-safe_log(fn_hmc, x) for x in xs_d])
            # Shift so minimum is 0
            U_vals -= U_vals.min()

            fig_u = make_subplots(rows=2, cols=1,
                                  subplot_titles=["Target π(q)  — the density",
                                                  "Potential U(q) = −log π(q)  — the landscape"],
                                  vertical_spacing=0.18)
            fig_u.add_trace(go.Scatter(x=xs_d, y=pd_d, fill="tozeroy", mode="none",
                                       fillcolor="rgba(0,204,150,0.12)",
                                       showlegend=False), 1, 1)
            fig_u.add_trace(go.Scatter(x=xs_d, y=pd_d, mode="lines",
                                       line=dict(color=C_GREEN, width=2.5),
                                       showlegend=False), 1, 1)
            fig_u.add_trace(go.Scatter(x=xs_d, y=U_vals, fill="tozeroy", mode="none",
                                       fillcolor="rgba(99,110,250,0.12)",
                                       showlegend=False), 2, 1)
            fig_u.add_trace(go.Scatter(x=xs_d, y=U_vals, mode="lines",
                                       line=dict(color=C_BLUE, width=2.5),
                                       showlegend=False), 2, 1)
            fig_u.update_layout(height=440, margin=dict(t=60, b=20))
            fig_u.update_yaxes(title_text="π(q)", row=1, col=1)
            fig_u.update_yaxes(title_text="U(q)", row=2, col=1)
            st.plotly_chart(fig_u, use_container_width=True)

    # ── HMC Step 2: Hamiltonian dynamics ──────────────────────────────────────
    elif hstep == 2:
        L, R = st.columns([1, 2])
        with L:
            st.markdown("## Hamiltonian dynamics")
            st.markdown("Add an auxiliary **momentum** p ~ N(0,1) to form the Hamiltonian:")
            st.latex(r"H(q,p) = \underbrace{-\log\pi(q)}_{U(q)} + \underbrace{\frac{p^2}{2}}_{K(p)}")
            st.markdown("""
The joint distribution of (q, p) is:
""")
            st.latex(r"\pi(q,p) \propto e^{-H(q,p)}")
            st.markdown("""
Marginalising over p recovers our original target π(q).

The **equations of motion** on constant-H surfaces are:
""")
            st.latex(r"\frac{dq}{dt} = p \qquad \frac{dp}{dt} = \nabla\!\log\pi(q)")
            st.markdown("""
The gradient ∇log π acts as a *restoring force*, pushing the ball toward high-density regions. Trajectories on constant-H surfaces explore the target **without random-walk diffusion**.
""")
        with R:
            qs_ph = np.linspace(*xr_hmc, 120)
            ps_ph = np.linspace(-4., 4., 120)
            U_g   = np.array([[-safe_log(fn_hmc, q) for q in qs_ph] for _ in ps_ph])
            _, PG = np.meshgrid(qs_ph, ps_ph)
            H_g   = U_g + 0.5 * PG**2

            fig_h2 = go.Figure()
            fig_h2.add_trace(go.Contour(x=qs_ph, y=ps_ph, z=H_g,
                                        colorscale="Viridis",
                                        contours=dict(showlabels=True,
                                                      labelfont=dict(size=10, color="white")),
                                        colorbar=dict(title="H(q,p)", len=0.6)))
            fig_h2.update_layout(
                title="H(q, p) contours — trajectories flow along these surfaces",
                xaxis_title="q  (position)", yaxis_title="p  (momentum)",
                height=440, margin=dict(t=60, b=20),
            )
            st.plotly_chart(fig_h2, use_container_width=True)
            st.caption("Each contour line is a constant-energy surface. HMC proposals travel *along* these lines.")

    # ── HMC Step 3: The leapfrog integrator ────────────────────────────────────
    elif hstep == 3:
        L, R = st.columns([1, 2])
        with L:
            st.markdown("## The leapfrog integrator")
            st.markdown("""
To simulate Hamilton's equations we use the **leapfrog** (velocity Verlet) scheme — the same symplectic integrator from numerical ODEs:
""")
            st.latex(r"""
\begin{aligned}
p_{1/2} &\leftarrow p + \tfrac{\varepsilon}{2}\nabla\!\log\pi(q) \\
q'      &\leftarrow q + \varepsilon\, p_{1/2} \\
p'      &\leftarrow p_{1/2} + \tfrac{\varepsilon}{2}\nabla\!\log\pi(q')
\end{aligned}
""")
            st.markdown("For L steps, adjacent half-steps fuse into full steps.")
            st.markdown("---")
            lv_q0  = st.slider("Start q₀", float(xr_hmc[0])*0.6, float(xr_hmc[1])*0.6,
                               1.5, 0.1, key="lv_q0h")
            lv_p0  = st.slider("Start p₀", -3.0, 3.0, 1.0, 0.1, key="lv_p0h")
            lv_eps = st.slider("Step size ε", 0.05, 0.6, 0.2, 0.05, key="lv_epsh")
            lv_L   = st.slider("Steps L", 3, 50, 20, 1, key="lv_Lh")
            st.caption("*Orange dashed = Euler reference (shown for comparison).*")
            st.markdown("*Press ▶ Play to watch the trajectory build up.*")
        with R:
            fig_lf = build_leapfrog_anim(hmc_tgt, lv_q0, lv_p0, lv_eps, lv_L)
            st.plotly_chart(fig_lf, use_container_width=True)

    # ── HMC Step 4: Euler vs leapfrog ──────────────────────────────────────────
    elif hstep == 4:
        L, R = st.columns([1, 2])
        with L:
            st.markdown("## Why leapfrog? Euler vs. leapfrog")
            st.markdown("""
Forward Euler applied to Hamilton's equations:
```
q' = q + ε·p
p' = p + ε·∇log π(q)
```
is **not symplectic** — it injects energy into the system on every step.

The leapfrog is **symplectic**: it exactly conserves a *modified* Hamiltonian H̃ = H + O(ε²), keeping the energy error **bounded for all time**.

| | Energy error | Long-time behavior |
|---|---|---|
| Euler | O(ε) per step | Grows without bound |
| Leapfrog | O(ε²) per step | Stays bounded |

In HMC: Euler's energy drift → ΔH is large → proposals rejected → poor mixing.
Leapfrog's near-conservation → ΔH ≈ 0 → nearly 100% acceptance.
""")
            st.info("This is why HMC uses leapfrog and not a standard ODE solver.")
        with R:
            lv2_q0  = st.slider("Start q₀", float(xr_hmc[0])*0.6, float(xr_hmc[1])*0.6,
                                1.5, 0.1, key="lv2_q0")
            lv2_p0  = st.slider("Start p₀", -3.0, 3.0, 1.0, 0.1, key="lv2_p0")
            lv2_eps = st.slider("Step size ε", 0.05, 0.8, 0.25, 0.05, key="lv2_eps")
            lv2_L   = st.slider("Steps L", 5, 60, 30, 1, key="lv2_L")

            _, _, traj_lf2 = leapfrog(lv2_q0, lv2_p0, fn_hmc, lv2_eps, lv2_L)
            traj_eu2       = euler_integration(lv2_q0, lv2_p0, fn_hmc, lv2_eps, lv2_L)

            pc1, pc2 = st.columns(2)
            with pc1:
                st.plotly_chart(make_phase_space(fn_hmc, xr_hmc, traj_lf2, traj_eu2,
                                                  "Phase space"),
                                use_container_width=True)
            with pc2:
                st.plotly_chart(make_hamiltonian_fig(traj_lf2, traj_eu2),
                                use_container_width=True)

            H0    = traj_lf2[0][2]
            dH_lf = abs(traj_lf2[-1][2] - H0)
            dH_eu = abs(traj_eu2[-1][2] - H0)
            m1, m2 = st.columns(2)
            m1.metric("Leapfrog |ΔH|", f"{dH_lf:.5f}")
            m2.metric("Euler |ΔH|", f"{dH_eu:.5f}",
                      delta=f"{dH_eu - dH_lf:+.5f} vs leapfrog",
                      delta_color="inverse")

    # ── HMC Step 5: HMC in action ──────────────────────────────────────────────
    elif hstep == 5:
        L, R = st.columns([1, 2])
        with L:
            st.markdown("## HMC in action")
            st.markdown("""
Each HMC iteration:
1. Draw fresh momentum **p ~ N(0,1)**
2. Run **L leapfrog steps** of size ε → proposed (q*, p*)
3. Accept with prob min(1, exp(−ΔH))

Because ΔH ≈ 0, acceptance is high and the chain **hops far** along the density — far less correlated than Metropolis.

Watch the red star jump across the density with each new sample.
""")
            st.markdown("---")
            h_eps  = st.slider("Step size ε", 0.05, 0.6, 0.2, 0.05, key="hmc_eps_a")
            h_L    = st.slider("Leapfrog steps L", 3, 40, 20, 1, key="hmc_L_a")
            h_n    = st.slider("Samples to animate", 30, 120, 70, 5, key="hmc_n_a")
            h_seed = st.number_input("Seed", 0, 999, 42, key="hmc_seed_a")
            st.markdown("*Press ▶ Play, or drag the slider.*")

            st.markdown("---")
            st.markdown("**Run full diagnostics**")
            h_n_full    = st.slider("Samples", 500, 8_000, 3_000, 500, key="hmc_nf")
            h_burn_full = st.slider("Burn-in", 100, 2_000, 500, 100, key="hmc_bf")
            h_seed_full = st.number_input("Seed", 0, 9_999, 42, key="hmc_sf")
            if st.button("Run full HMC", type="primary", key="run_hmc_f"):
                with st.spinner("Running HMC…"):
                    sh, rh, lt = _cached_hmc(hmc_tgt, h_n_full, h_burn_full,
                                             h_eps, h_L, int(h_seed_full))
                st.session_state["hmc_res"] = dict(s=sh, r=rh, tgt=hmc_tgt,
                                                    eps=h_eps, L=h_L, lt=lt)
        with R:
            fig_ha = build_hmc_samples_anim(hmc_tgt, int(h_n), h_eps, h_L, int(h_seed))
            st.plotly_chart(fig_ha, use_container_width=True)

            res_h = st.session_state["hmc_res"]
            if res_h:
                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("Acceptance rate", f"{res_h['r']:.1%}")
                c2.metric("Samples", f"{res_h['s'].shape[0]:,}")
                c3.metric("ε·L", f"{res_h['eps'] * res_h['L']:.2f}")
                st.plotly_chart(
                    make_diagnostics(res_h["s"], TARGETS[res_h["tgt"]]["fn"],
                                     TARGETS[res_h["tgt"]]["range"],
                                     C_RED, f"HMC — {res_h['tgt']}"),
                    use_container_width=True,
                )
                if res_h["lt"]:
                    st.markdown("**Last leapfrog trajectory**")
                    st.plotly_chart(
                        make_phase_space(fn_hmc, xr_hmc, res_h["lt"],
                                         title="Last HMC leapfrog step"),
                        use_container_width=True,
                    )

# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TAB
# ─────────────────────────────────────────────────────────────────────────────
with tab_cmp:
    st.header("Metropolis-Hastings vs. HMC — side-by-side")
    st.markdown("""
Run both algorithms on the same target with matched sample counts.
The **ACF overlay** at the bottom is the key diagnostic: faster decay → better mixing.
    """)

    cc_l, cc_r = st.columns([1, 3])
    with cc_l:
        cmp_tgt  = st.selectbox("Target", list(TARGETS.keys()), key="cmp_tgt")
        cmp_n    = st.slider("Samples each", 500, 8_000, 3_000, 500, key="cmp_n")
        cmp_burn = st.slider("Burn-in each", 100, 2_000, 500, 100, key="cmp_burn")
        cmp_seed = st.number_input("Seed", 0, 9_999, 42, key="cmp_seed")
        st.markdown("**Metropolis**")
        cmp_std  = st.slider("Proposal σ", 0.05, 5.0, 1.0, 0.05, key="cmp_std")
        st.markdown("**HMC**")
        cmp_eps  = st.slider("Step size ε", 0.05, 0.6, 0.2, 0.05, key="cmp_eps")
        cmp_L    = st.slider("Leapfrog steps L", 1, 50, 20, 1, key="cmp_L")
        run_cmp  = st.button("Run comparison", type="primary", key="run_cmp")

    with cc_r:
        if run_cmp:
            t_c = TARGETS[cmp_tgt]
            with st.spinner("Running both samplers…"):
                mh_s, mh_r = _cached_metro(cmp_tgt, cmp_n, cmp_burn, cmp_std, int(cmp_seed))
                hmc_s, hmc_r, _ = _cached_hmc(cmp_tgt, cmp_n, cmp_burn,
                                               cmp_eps, cmp_L, int(cmp_seed))

            acf_mh  = compute_acf(mh_s)
            acf_hmc = compute_acf(hmc_s)

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("MH acceptance",  f"{mh_r:.1%}")
            r2.metric("HMC acceptance", f"{hmc_r:.1%}")
            r3.metric("MH ACF lag-1",   f"{acf_mh[0]:.3f}")
            r4.metric("HMC ACF lag-1",  f"{acf_hmc[0]:.3f}",
                      delta=f"{acf_hmc[0] - acf_mh[0]:+.3f} vs MH",
                      delta_color="inverse")

            st.plotly_chart(make_diagnostics(mh_s, t_c["fn"], t_c["range"],
                                              C_BLUE, f"Metropolis — {cmp_tgt}"),
                            use_container_width=True)
            st.plotly_chart(make_diagnostics(hmc_s, t_c["fn"], t_c["range"],
                                              C_RED, f"HMC — {cmp_tgt}"),
                            use_container_width=True)

            fig_acf = go.Figure()
            fig_acf.add_trace(go.Scatter(x=list(range(1, len(acf_mh)+1)), y=acf_mh,
                                          mode="lines", name="Metropolis",
                                          line=dict(color=C_BLUE, width=2.5)))
            fig_acf.add_trace(go.Scatter(x=list(range(1, len(acf_hmc)+1)), y=acf_hmc,
                                          mode="lines", name="HMC",
                                          line=dict(color=C_RED, width=2.5)))
            fig_acf.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_acf.update_layout(title="ACF comparison — faster decay = better mixing",
                                   xaxis_title="Lag", yaxis_title="Autocorrelation",
                                   height=300, margin=dict(t=50, b=20))
            st.plotly_chart(fig_acf, use_container_width=True)
        else:
            st.info("Configure both samplers and click **Run comparison**.")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD A PPL TAB
# ─────────────────────────────────────────────────────────────────────────────
with tab_ppl:
    st.header("Build a Probabilistic Programming Language from Scratch")
    st.markdown("""
We'll construct a minimal PPL step-by-step — distributions, variable nodes,
a model graph, and an MCMC sampler — then run it on a realistic hierarchical model.
All code lives in **`ppl.py`** alongside this app.
""")

    PPL_LABELS = [
        "Distributions",
        "Variable nodes & Model",
        "Metropolis sampler",
        "HMC in the PPL",
        "Random effects demo",
    ]
    ppl_step = step_nav("ppl_step", PPL_LABELS)

    # ── PPL Step 0: Distributions ──────────────────────────────────────────────
    if ppl_step == 0:
        L, R = st.columns([1, 1])
        with L:
            st.markdown("## Distributions")
            st.markdown("""
Every variable in a probabilistic model has a **distribution** — a
probability density that tells us how plausible each value is.
In our PPL, each distribution class just needs one method: `log_prob(x)`.

We use **log**-probabilities everywhere because:
- They avoid numerical underflow with products of small probabilities
- They turn products (joint distributions) into sums
- MCMC only needs *differences* of log-probabilities, so constants cancel
""")
            st.code("""\
import scipy.stats as sp
import numpy as np

class Normal:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean   # can be a callable for linking
        self.std  = std

    def log_prob(self, x):
        mean = self.mean() if callable(self.mean) else self.mean
        std  = self.std()  if callable(self.std)  else self.std
        return float(sp.norm.logpdf(x, mean, std))

class HalfNormal:          # positive-only — great prior for variances
    def __init__(self, scale=1.0):
        self.scale = scale
    def log_prob(self, x):
        if x < 0: return -np.inf
        return float(sp.halfnorm.logpdf(x, scale=self.scale))

class Gamma:               # rate parameterisation: scale = 1/beta
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha; self.beta = beta
    def log_prob(self, x):
        if x <= 0: return -np.inf
        return float(sp.gamma.logpdf(x, a=self.alpha, scale=1/self.beta))

class Beta:                # supported on (0, 1)
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha; self.beta = beta
    def log_prob(self, x):
        if x <= 0 or x >= 1: return -np.inf
        return float(sp.beta.logpdf(x, self.alpha, self.beta))
""", language="python")

        with R:
            st.markdown("### Interactive: explore the distributions")
            dist_choice = st.selectbox("Distribution", ["Normal", "HalfNormal", "Gamma", "Beta"],
                                       key="ppl_dist")
            xs_d = np.linspace(-5, 5, 400)
            if dist_choice == "Normal":
                mn = st.slider("mean", -3.0, 3.0, 0.0, 0.1, key="ppl_mn")
                sd = st.slider("std",   0.1, 3.0, 1.0, 0.1, key="ppl_sd")
                xs_d = np.linspace(mn - 4*sd, mn + 4*sd, 400)
                dist_obj = _ppl.Normal(mn, sd)
            elif dist_choice == "HalfNormal":
                sc = st.slider("scale", 0.2, 3.0, 1.0, 0.1, key="ppl_sc")
                xs_d = np.linspace(0, sc*4, 400)
                dist_obj = _ppl.HalfNormal(sc)
            elif dist_choice == "Gamma":
                al = st.slider("alpha (shape)", 0.5, 8.0, 2.0, 0.5, key="ppl_al")
                be = st.slider("beta (rate)",   0.2, 5.0, 1.0, 0.2, key="ppl_be")
                xs_d = np.linspace(0.01, al/be + 5*np.sqrt(al)/be, 400)
                dist_obj = _ppl.Gamma(al, be)
            else:  # Beta
                a2 = st.slider("alpha", 0.5, 8.0, 2.0, 0.5, key="ppl_a2")
                b2 = st.slider("beta",  0.5, 8.0, 5.0, 0.5, key="ppl_b2")
                xs_d = np.linspace(0.001, 0.999, 400)
                dist_obj = _ppl.Beta(a2, b2)

            lps = np.array([dist_obj.log_prob(x) for x in xs_d])
            pd_vals = np.exp(np.where(np.isfinite(lps), lps, -30))

            fig_dp = go.Figure()
            fig_dp.add_trace(go.Scatter(x=xs_d, y=pd_vals, fill="tozeroy", mode="none",
                                        fillcolor="rgba(99,110,250,0.15)", showlegend=False))
            fig_dp.add_trace(go.Scatter(x=xs_d, y=pd_vals, mode="lines",
                                        line=dict(color=C_BLUE, width=2.5), name=dist_choice))
            fig_dp.update_layout(title=f"{dist_choice} PDF", xaxis_title="x",
                                  yaxis_title="density", height=340,
                                  margin=dict(t=50, b=20))
            st.plotly_chart(fig_dp, use_container_width=True)

            st.markdown(f"**log_prob at x = 0.5:** `{dist_obj.log_prob(0.5):.4f}`")
            st.info("Click **Next ▶** to see how distributions attach to variables in a model graph.")

    # ── PPL Step 1: Variable nodes & Model ────────────────────────────────────
    elif ppl_step == 1:
        L, R = st.columns([1, 1])
        with L:
            st.markdown("## Variable nodes & the Model graph")
            st.markdown("""
A **Variable** is a named node in the graph. It stores:
- which distribution it was drawn from
- its parent nodes (dependencies)
- whether it's **observed** (data — fixed value) or **latent** (unknown)

A **Deterministic** node is a pure function of parents — it has no log_prob
contribution, it just computes a derived quantity.

The **Model** class stitches nodes together and can evaluate the
**joint log-probability** — the sum of all node log-probs in topological order.
This is the core quantity MCMC needs; the normalising constant never appears.
""")
            st.code("""\
class Variable:
    def __init__(self, name, dist, parents=None,
                 observed=False, observed_data=None):
        self.name          = name
        self.dist          = dist
        self.parents       = parents or []
        self.observed      = observed
        self.value         = observed_data
        self.deterministic = isinstance(dist, Deterministic)

class Model:
    def __init__(self):
        self.variables = {}

    def add_variable(self, name, dist, parents=None,
                     observed=False, observed_data=None):
        var = Variable(name, dist, parents, observed, observed_data)
        self.variables[name] = var
        return var

    def add_deterministic(self, name, fn, parents):
        var = Variable(name, Deterministic(fn), parents=parents)
        self.variables[name] = var
        return var

    def log_prob(self, state: dict) -> float:
        logp = 0.0
        for var in self._topological_sort():
            if var.deterministic:
                var.value = var.dist.evaluate()
            else:
                if not var.observed:
                    var.value = state[var.name]
                logp += var.dist.log_prob(var.value)
        return logp
""", language="python")

        with R:
            st.markdown("### Live demo: build and evaluate a model")
            st.markdown("""
Let's build a simple Bayesian normal model:
```
mu    ~ Normal(0, 5)      # prior on the mean
sigma ~ HalfNormal(1)     # prior on the SD (must be positive)
y_i   ~ Normal(mu, sigma) # likelihood for each observation
```
""")
            demo_mu    = st.slider("Try state: mu",    -3.0, 3.0, 1.0, 0.1, key="ppl_dmu")
            demo_sigma = st.slider("Try state: sigma",  0.1, 3.0, 1.0, 0.1, key="ppl_dsg")
            demo_n     = st.slider("# observations",    3, 20, 8, 1, key="ppl_dn")

            np.random.seed(42)
            y_demo = np.random.normal(1.5, 0.8, demo_n)

            m_demo   = _ppl.Model()
            mu_d     = m_demo.add_variable("mu",    _ppl.Normal(0, 5))
            sigma_d  = m_demo.add_variable("sigma", _ppl.HalfNormal(1))

            class _NormLik:
                def __init__(self, mu_v, sig_v, y):
                    self.mu_v = mu_v; self.sig_v = sig_v; self.y = y
                def log_prob(self, x):
                    return float(np.sum(stats.norm.logpdf(self.y, self.mu_v.value, self.sig_v.value)))

            m_demo.add_variable("y", _NormLik(mu_d, sigma_d, y_demo),
                                parents=[mu_d, sigma_d], observed=True, observed_data=y_demo)

            state_demo = {"mu": demo_mu, "sigma": demo_sigma}
            lp_val     = m_demo.log_prob(state_demo)

            logp_prior_mu    = _ppl.Normal(0, 5).log_prob(demo_mu)
            logp_prior_sigma = _ppl.HalfNormal(1).log_prob(demo_sigma)
            logp_lik         = float(np.sum(stats.norm.logpdf(y_demo, demo_mu, demo_sigma)))

            st.markdown(f"""
**Evaluating at state** `{{mu={demo_mu:.1f}, sigma={demo_sigma:.1f}}}`

| Component | Value |
|---|---|
| log p(mu) | `{logp_prior_mu:.3f}` |
| log p(sigma) | `{logp_prior_sigma:.3f}` |
| log p(y \| mu, sigma) | `{logp_lik:.3f}` |
| **Joint log_prob** | **`{lp_val:.3f}`** |

MCMC uses differences of joint log_prob values — the normalising constant Z cancels.
""")
            st.info("Move the sliders: joint log_prob increases as you approach the true mean (1.5).")

    # ── PPL Step 2: Metropolis sampler ────────────────────────────────────────
    elif ppl_step == 2:
        L, R = st.columns([1, 1])
        with L:
            st.markdown("## The Metropolis sampler")
            st.markdown("""
The MCMC class wraps a Model and samples from its posterior.
At each step it proposes a new state by perturbing every free variable
independently with a Gaussian of width `proposal_std`, then accepts or
rejects via the log-acceptance ratio.
""")
            st.code("""\
class MCMC:
    def __init__(self, model, initial_state=None, proposal_std=0.1):
        self.model        = model
        self.proposal_std = proposal_std
        self._init_state  = initial_state or {k: 0.0 for k in model.free_vars}

    def _reset(self):
        self.current_state = dict(self._init_state)
        self.chain = []; self.accepted = 0; self.proposed = 0

    @property
    def acceptance_rate(self):
        return self.accepted / self.proposed if self.proposed > 0 else 0.0

    def _proposal_step(self):
        return {k: np.random.normal(self.current_state[k], self.proposal_std)
                for k in self.model.free_vars}

    def _metropolis(self, n_samples, burn_in):
        for i in range(n_samples + burn_in):
            proposed = self._proposal_step()
            log_alpha = (self.model.log_prob(proposed)
                         - self.model.log_prob(self.current_state))
            if np.log(np.random.rand()) < log_alpha:
                self.current_state = proposed
                if i >= burn_in: self.accepted += 1
            self.proposed += 1
            if i >= burn_in: self.chain.append(self.current_state.copy())

    def sample(self, method='metropolis', n_samples=1000, burn_in=500, **kw):
        self._reset()
        if method == 'metropolis':
            self._metropolis(n_samples, burn_in)
        elif method == 'hmc':
            self._hmc(n_samples, burn_in, **kw)
        return self.chain
""", language="python")

        with R:
            st.markdown("### Interactive playground")
            ppl_tgt_name = st.selectbox(
                "Target", ["Standard Normal", "Bimodal Mixture", "Student-t (df=3)"],
                key="ppl_play_tgt"
            )
            ppl_pstd = st.slider("proposal_std", 0.05, 4.0, 0.8, 0.05, key="ppl_pstd")
            ppl_nsamp = st.slider("n_samples", 500, 5000, 2000, 500, key="ppl_nsamp")
            ppl_seed  = st.number_input("Seed", 0, 999, 7, key="ppl_seed2")

            if st.button("Run Metropolis", type="primary", key="ppl_run_mh"):
                tgt_fn = TARGETS[ppl_tgt_name]["fn"]
                class _Dist:
                    def __init__(self, fn): self.fn = fn
                    def log_prob(self, x): return float(self.fn(x))
                np.random.seed(int(ppl_seed))
                m_p  = _ppl.Model()
                m_p.add_variable("x", _Dist(tgt_fn))
                mcp  = _ppl.MCMC(m_p, initial_state={"x": 0.0}, proposal_std=ppl_pstd)
                with st.spinner("Sampling…"):
                    chain_p = mcp.sample("metropolis", n_samples=int(ppl_nsamp), burn_in=500)
                xs_p = np.array([s["x"] for s in chain_p])
                st.session_state["ppl_samples"] = dict(xs=xs_p, rate=mcp.acceptance_rate,
                                                        tgt=ppl_tgt_name, method="Metropolis")

            res_p = st.session_state["ppl_samples"]
            if res_p:
                st.metric("Acceptance rate", f"{res_p['rate']:.1%}")
                xs_p = res_p["xs"]
                xr_p = TARGETS[res_p["tgt"]]["range"]
                st.plotly_chart(
                    make_diagnostics(xs_p, TARGETS[res_p["tgt"]]["fn"], xr_p,
                                     C_BLUE, f"PPL Metropolis — {res_p['tgt']}"),
                    use_container_width=True,
                )

    # ── PPL Step 3: HMC ───────────────────────────────────────────────────────
    elif ppl_step == 3:
        L, R = st.columns([1, 1])
        with L:
            st.markdown("## HMC in the PPL")
            st.markdown("""
HMC needs the **gradient** of log_prob with respect to the free variables.
Production PPLs (PyMC, Stan, Pyro) compute this via symbolic or algorithmic
differentiation.  Our PPL uses **central finite differences** — simple,
works with any distribution class, correct to O(h²).
""")
            st.code("""\
def _grad_log_prob(self, state, h=1e-4):
    \"\"\"Numerical gradient via central finite differences.\"\"\"
    g = {}
    for k in self.model.free_vars:
        s_p = {**state, k: state[k] + h}
        s_m = {**state, k: state[k] - h}
        g[k] = (self.model.log_prob(s_p)
                - self.model.log_prob(s_m)) / (2 * h)
    return g

def _hmc(self, n_samples, burn_in, step_size=0.05, n_leapfrog_steps=20):
    keys = self.model.free_vars
    for i in range(n_samples + burn_in):
        q = dict(self.current_state)
        p = {k: float(np.random.standard_normal()) for k in keys}

        # Half-step momentum, then L full leapfrog steps
        g    = self._grad_log_prob(q)
        p_hf = {k: p[k] + 0.5 * step_size * g[k] for k in keys}
        q_new, p_new = dict(q), dict(p_hf)
        for l in range(n_leapfrog_steps):
            q_new = {k: q_new[k] + step_size * p_new[k] for k in keys}
            g_new = self._grad_log_prob(q_new)
            factor = 0.5 if l == n_leapfrog_steps - 1 else 1.0
            p_new  = {k: p_new[k] + factor * step_size * g_new[k] for k in keys}

        H_curr = (-self.model.log_prob(q)
                  + 0.5 * sum(p[k]**2 for k in keys))
        H_prop = (-self.model.log_prob(q_new)
                  + 0.5 * sum(p_new[k]**2 for k in keys))

        if np.log(np.random.rand()) < H_curr - H_prop:
            self.current_state = q_new
            if i >= burn_in: self.accepted += 1
        self.proposed += 1
        if i >= burn_in:
            self.chain.append(self.current_state.copy())
""", language="python")

            st.info("""
**Why not autograd / JAX here?**
The distribution classes call `scipy.stats`, which isn't differentiable by
autograd.  A production PPL rewrites its math in a differentiable backend
(pytensor, torch, jax) from the start.  Our numerical gradients are exact
to O(h²) and completely general.
""")

        with R:
            st.markdown("### Compare Metropolis vs HMC on the same target")
            ppl_tgt_h = st.selectbox(
                "Target", ["Standard Normal", "Bimodal Mixture", "Student-t (df=3)"],
                key="ppl_hmc_tgt"
            )
            p_std_h = st.slider("Metropolis proposal_std", 0.1, 3.0, 0.8, 0.1, key="ppl_hstd")
            h_ss    = st.slider("HMC step_size",         0.05, 0.6, 0.2, 0.05, key="ppl_hss")
            h_L     = st.slider("HMC leapfrog steps",    3, 30, 15, 1, key="ppl_hL")
            h_n     = st.slider("n_samples each",        500, 3000, 1500, 500, key="ppl_hn")
            h_seed  = st.number_input("Seed", 0, 999, 7, key="ppl_hsd")

            if st.button("Run both", type="primary", key="ppl_run_both"):
                tgt_fn2 = TARGETS[ppl_tgt_h]["fn"]
                class _Dist2:
                    def __init__(self, fn): self.fn = fn
                    def log_prob(self, x): return float(self.fn(x))
                np.random.seed(int(h_seed))
                # Metropolis
                m_mh2 = _ppl.Model(); m_mh2.add_variable("x", _Dist2(tgt_fn2))
                mc_mh = _ppl.MCMC(m_mh2, {"x": 0.0}, p_std_h)
                with st.spinner("Metropolis…"):
                    ch_mh = mc_mh.sample("metropolis", int(h_n), 500)
                # HMC
                np.random.seed(int(h_seed))
                m_hm2 = _ppl.Model(); m_hm2.add_variable("x", _Dist2(tgt_fn2))
                mc_hm = _ppl.MCMC(m_hm2, {"x": 0.0}, 0.1)
                with st.spinner("HMC…"):
                    ch_hm = mc_hm.sample("hmc", int(h_n), 500,
                                          step_size=h_ss, n_leapfrog_steps=int(h_L))
                xs_mh = np.array([s["x"] for s in ch_mh])
                xs_hm = np.array([s["x"] for s in ch_hm])
                st.session_state["ppl_samples"] = dict(xs_mh=xs_mh, xs_hm=xs_hm,
                                                        mh_rate=mc_mh.acceptance_rate,
                                                        hm_rate=mc_hm.acceptance_rate,
                                                        tgt=ppl_tgt_h, mode="compare")

            res_h = st.session_state["ppl_samples"]
            if res_h and res_h.get("mode") == "compare":
                c1, c2 = st.columns(2)
                c1.metric("MH acceptance",  f"{res_h['mh_rate']:.1%}")
                c2.metric("HMC acceptance", f"{res_h['hm_rate']:.1%}")
                acf_mh2 = compute_acf(res_h["xs_mh"])
                acf_hm2 = compute_acf(res_h["xs_hm"])
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Scatter(x=list(range(1, len(acf_mh2)+1)), y=acf_mh2,
                                             mode="lines", name="Metropolis",
                                             line=dict(color=C_BLUE, width=2)))
                fig_cmp.add_trace(go.Scatter(x=list(range(1, len(acf_hm2)+1)), y=acf_hm2,
                                             mode="lines", name="HMC",
                                             line=dict(color=C_RED, width=2)))
                fig_cmp.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_cmp.update_layout(title="ACF — PPL Metropolis vs PPL HMC",
                                       xaxis_title="Lag", height=280, margin=dict(t=50, b=20))
                st.plotly_chart(fig_cmp, use_container_width=True)

                tgt_fn3 = TARGETS[res_h["tgt"]]["fn"]
                xr3 = TARGETS[res_h["tgt"]]["range"]
                xs3, pd3 = density_curve(tgt_fn3, xr3)
                fig_ov = go.Figure()
                fig_ov.add_trace(go.Histogram(x=res_h["xs_mh"], histnorm="probability density",
                                               nbinsx=60, opacity=0.45, name="Metropolis",
                                               marker_color=C_BLUE))
                fig_ov.add_trace(go.Histogram(x=res_h["xs_hm"], histnorm="probability density",
                                               nbinsx=60, opacity=0.45, name="HMC",
                                               marker_color=C_RED))
                fig_ov.add_trace(go.Scatter(x=xs3, y=pd3, mode="lines",
                                             line=dict(color=C_GREEN, width=2.5), name="True π"))
                fig_ov.update_layout(barmode="overlay", title="Sample histograms vs target",
                                      height=280, margin=dict(t=50, b=20))
                st.plotly_chart(fig_ov, use_container_width=True)

    # ── PPL Step 4: Random effects demo ───────────────────────────────────────
    elif ppl_step == 4:
        st.markdown("## Full demo — random effects model")
        st.markdown(r"""
We fit a hierarchical model to simulated classroom data:
$$y_{ij} = \mu + u_i + \varepsilon_{ij}$$
- **μ** — global mean (prior: N(0, 5))
- **u_i** — group-specific random effects (prior: N(0, σ_u²))
- **σ_u** — between-group SD (prior: HalfNormal(1))
- **σ_e** — within-group SD (prior: HalfNormal(1))
- **y_ij** — observed, 5 groups × 20 observations
""")

        code_col, ctrl_col = st.columns([3, 1])
        with ctrl_col:
            re_nsamp  = st.slider("n_samples", 1000, 6000, 3000, 500, key="ppl_re_n")
            re_burn   = st.slider("burn_in",    500, 2000, 1000, 500, key="ppl_re_b")
            re_pstd   = st.slider("proposal_std", 0.02, 0.3, 0.08, 0.01, key="ppl_re_std")
            run_re    = st.button("Run sampler", type="primary", key="ppl_run_re")

        with code_col:
            st.code("""\
# ── Simulate data ─────────────────────────────────────────────────────────────
np.random.seed(42)
n_groups, n_per = 5, 20
true_mu      = 2.0
true_sigma_u = 1.5   # between-group SD
true_sigma_e = 0.5   # within-group SD
group_ids    = np.repeat(np.arange(n_groups), n_per)
true_u       = np.random.normal(0, true_sigma_u, n_groups)
y_obs        = (true_mu + true_u[group_ids]
                + np.random.normal(0, true_sigma_e, n_groups * n_per))

# ── Build model ───────────────────────────────────────────────────────────────
model   = Model()
mu      = model.add_variable('mu',      Normal(0, 5))
sigma_u = model.add_variable('sigma_u', HalfNormal(1))
sigma_e = model.add_variable('sigma_e', HalfNormal(1))
u       = [model.add_variable(f'u_{i}', Normal(0, 1), parents=[sigma_u])
           for i in range(n_groups)]

# NormalVecLikelihood scores all observations at once
mu_obs  = model.add_deterministic(
    'mu_obs',
    lambda: mu.value + np.array([u[g].value for g in group_ids]),
    parents=[mu, *u])
y = model.add_variable('y',
    NormalVecLikelihood(mu_obs, sigma_e, y_obs),
    parents=[mu_obs, sigma_e], observed=True)

# ── Sample ────────────────────────────────────────────────────────────────────
init = {'mu': 0.0, 'sigma_u': 1.0, 'sigma_e': 1.0,
        **{f'u_{i}': 0.0 for i in range(n_groups)}}
mcmc = MCMC(model, initial_state=init, proposal_std=0.08)
samples = mcmc.sample('metropolis', n_samples=3000, burn_in=1000)
""", language="python")

        if run_re:
            np.random.seed(42)
            n_groups, n_per = 5, 20
            true_mu_re      = 2.0
            true_sigma_u_re = 1.5
            true_sigma_e_re = 0.5
            group_ids_re    = np.repeat(np.arange(n_groups), n_per)
            true_u_re       = np.random.normal(0, true_sigma_u_re, n_groups)
            y_obs_re        = (true_mu_re + true_u_re[group_ids_re]
                               + np.random.normal(0, true_sigma_e_re, n_groups * n_per))

            model_re = _ppl.Model()
            mu_re    = model_re.add_variable("mu",      _ppl.Normal(0, 5))
            su_re    = model_re.add_variable("sigma_u", _ppl.HalfNormal(1))
            se_re    = model_re.add_variable("sigma_e", _ppl.HalfNormal(1))
            u_re     = [model_re.add_variable(f"u_{i}", _ppl.Normal(0, 1), parents=[su_re])
                        for i in range(n_groups)]
            mu_obs_re = model_re.add_deterministic(
                "mu_obs",
                lambda: mu_re.value + np.array([u_re[g].value for g in group_ids_re]),
                parents=[mu_re, *u_re])
            model_re.add_variable("y", _ppl.NormalVecLikelihood(mu_obs_re, se_re, y_obs_re),
                                   parents=[mu_obs_re, se_re], observed=True, observed_data=y_obs_re)

            init_re = {"mu": 0.0, "sigma_u": 1.0, "sigma_e": 1.0,
                       **{f"u_{i}": 0.0 for i in range(n_groups)}}
            mc_re = _ppl.MCMC(model_re, initial_state=init_re, proposal_std=float(re_pstd))
            with st.spinner(f"Running Metropolis ({int(re_nsamp)} samples, burn-in {int(re_burn)})…"):
                samples_re = mc_re.sample("metropolis", int(re_nsamp), int(re_burn))
            st.session_state["ppl_samples"] = dict(
                mode="re",
                samples=samples_re,
                rate=mc_re.acceptance_rate,
                true_mu=true_mu_re, true_su=true_sigma_u_re, true_se=true_sigma_e_re,
                true_u=true_u_re, n_groups=n_groups,
            )

        res_re = st.session_state["ppl_samples"]
        if res_re and res_re.get("mode") == "re":
            st.metric("Acceptance rate", f"{res_re['rate']:.1%}")
            samps = res_re["samples"]
            params_re = {
                "mu":      (np.array([s["mu"]      for s in samps]), res_re["true_mu"],  C_BLUE),
                "sigma_u": (np.array([s["sigma_u"] for s in samps]), res_re["true_su"],  C_ORANGE),
                "sigma_e": (np.array([s["sigma_e"] for s in samps]), res_re["true_se"],  C_GREEN),
            }
            ng = res_re["n_groups"]
            u_chains_re = [np.array([s[f"u_{i}"] for s in samps]) for i in range(ng)]

            fig_re, axes_re = plt.subplots(3, 3, figsize=(13, 9))
            fig_re.suptitle("Random Effects Model — Posterior Diagnostics",
                             fontsize=13, fontweight="bold")
            colors_re = [C_BLUE, C_ORANGE, C_GREEN]
            for col, (name, (chain, true_val, col_c)) in enumerate(params_re.items()):
                hex_c = col_c
                axes_re[0, col].plot(chain, lw=0.5, color=hex_c, alpha=0.8)
                axes_re[0, col].axhline(true_val, color="red", lw=1.2, linestyle="--", label="true")
                axes_re[0, col].set_title(f"{name} — trace", fontsize=9)
                axes_re[0, col].legend(fontsize=7)

                axes_re[1, col].hist(chain, bins=50, color=hex_c, alpha=0.7, density=True)
                axes_re[1, col].axvline(true_val,       color="red",   lw=1.5, ls="--", label="true")
                axes_re[1, col].axvline(chain.mean(),   color="black", lw=1.5, ls="-",
                                         label=f"mean={chain.mean():.2f}")
                axes_re[1, col].set_title(f"{name} — posterior", fontsize=9)
                axes_re[1, col].legend(fontsize=7)

                acf_re = compute_acf(chain, 40)
                axes_re[2, col].bar(range(1, 41), acf_re, color=hex_c, alpha=0.7, width=0.8)
                axes_re[2, col].axhline(0, color="black", lw=0.8)
                axes_re[2, col].set_title(f"{name} — ACF", fontsize=9)
                axes_re[2, col].set_ylim(-0.3, 1.0)

            plt.tight_layout()
            st.pyplot(fig_re)
            plt.close(fig_re)

            # Random effects recovery
            fig_u2, ax_u2 = plt.subplots(figsize=(8, 4))
            post_means_u = [u_chains_re[i].mean() for i in range(ng)]
            post_stds_u  = [u_chains_re[i].std()  for i in range(ng)]
            x_u = np.arange(ng)
            ax_u2.errorbar(x_u, post_means_u, yerr=1.96*np.array(post_stds_u),
                           fmt="o", color=C_BLUE, capsize=6, label="posterior mean ± 1.96 SD")
            ax_u2.scatter(x_u, res_re["true_u"], color="red", zorder=5,
                          marker="x", s=90, lw=2, label="true uᵢ")
            ax_u2.axhline(0, color="gray", lw=0.8, ls="--")
            ax_u2.set_xticks(x_u); ax_u2.set_xticklabels([f"Group {i}" for i in range(ng)])
            ax_u2.set_title("Random effects recovery — posterior vs truth", fontsize=11)
            ax_u2.legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_u2)
            plt.close(fig_u2)


# ─────────────────────────────────────────────────────────────────────────────
# WILDFLOWER PHENOLOGY TAB
# ─────────────────────────────────────────────────────────────────────────────
with tab_pheno:
    st.header("Case Study — Wildflower Phenology")
    st.markdown("""
Different wildflower species need different amounts of accumulated warmth before they bloom.
We'll fit a **growing degree day (GDD) model** to 40 years of synthetic bloom observations,
using **(a) our own PPL** and **(b) PyMC with NUTS**, then compare how well each sampler
handles the correlated parameter posterior.
""")

    # ── Load data once ─────────────────────────────────────────────────────────
    T_daily_ph, doy_obs_ph, years_ph, true_params_ph = generate_phenology_data()
    n_years_ph = len(years_ph)

    # Shared colour palette — used in both PPL and PyMC sections
    ph_colors = [C_BLUE, C_RED, C_ORANGE, C_GREEN]

    # ── Section 1: Physical model ──────────────────────────────────────────────
    with st.expander("📐 Section 1 — The physical model", expanded=True):
        s1L, s1R = st.columns([1, 1])
        with s1L:
            st.markdown("### The GDD model")
            st.latex(r"\frac{dH}{dt} = \max\!\bigl(T(t) - T_{\text{base}},\; 0\bigr)")
            st.markdown("""
The plant accumulates **heat units** (growing degree days, GDD) on every day
the temperature exceeds a base threshold $T_{\\text{base}}$.
Bloom is triggered on the first day $t^*$ when total heat crosses the
species-specific threshold $H^*$:
""")
            st.latex(r"t^* = \min\bigl\{\,t : H(t) \ge H^*\bigr\}")
            st.markdown("""
We observe $t^*$ each year with a small amount of noise:
""")
            st.latex(r"\text{DOY}_y \;\sim\; \mathcal{N}\!\bigl(t^*_y(\theta),\;\sigma\bigr)")
            st.markdown("""
**Key point — threshold-crossing likelihood.**
Unlike trajectory-fitting models (where you compare a whole curve),
here the ODE produces **one number per year**: the day the heat crosses
the threshold. The likelihood compares that scalar prediction to the
observed first-bloom day. This structure is common in phenology, hydrology
(streamflow timing), and any model where what you observe is an *event*.
""")
            st.markdown("### Parameters")
            st.markdown("""
| Parameter | Meaning | Prior |
|---|---|---|
| **T_base** | Base temperature below which no development occurs (°C) | Normal(2, 3) |
| **H★** | Total heat units required to bloom (GDD) | HalfNormal(200) |
| **t₀** | Day of year to start accumulating heat | Uniform(1, 120) |
| **σ** | Observation noise (days) | HalfNormal(10) |
""")

        with s1R:
            st.markdown("### Example: year 1980")
            # Build example figure
            tp = true_params_ph
            T_ex = T_daily_ph[0]
            gdd_ex = np.maximum(T_ex - tp["T_base"], 0.0)
            gdd_ex[:int(tp["t0"])] = 0.0
            H_cum_ex = np.cumsum(gdd_ex)
            bloom_day_ex = int(np.argmax(H_cum_ex >= tp["H_star"]) + 1)

            fig_ode_ph, (ax_t_ph, ax_h_ph) = plt.subplots(
                2, 1, figsize=(7, 5), sharex=True)

            days_plot = np.arange(1, 181)   # show first 180 days
            ax_t_ph.plot(days_plot, T_ex[:180], color="steelblue", lw=1.5, label="T(t)")
            ax_t_ph.axhline(tp["T_base"], color="gray", lw=1.2, ls="--",
                            label=f"T_base = {tp['T_base']}°C")
            ax_t_ph.axvline(tp["t0"], color="darkorange", lw=1.5, ls="--",
                            label=f"t₀ = day {int(tp['t0'])}")
            ax_t_ph.fill_between(days_plot,
                                 tp["T_base"],
                                 np.maximum(T_ex[:180], tp["T_base"]),
                                 alpha=0.25, color="tomato", label="Daily GDD")
            ax_t_ph.set_ylabel("Temperature (°C)", fontsize=9)
            ax_t_ph.legend(fontsize=8, loc="upper left")
            ax_t_ph.set_title("Step 1 — Daily temperature (1980)", fontsize=10)

            ax_h_ph.plot(days_plot, H_cum_ex[:180], color="tomato", lw=2,
                         label="H(t) — accumulated GDD")
            ax_h_ph.axhline(tp["H_star"], color="green", lw=1.5, ls="--",
                            label=f"H★ = {tp['H_star']} GDD")
            ax_h_ph.axvline(tp["t0"], color="darkorange", lw=1.5, ls="--")
            if bloom_day_ex <= 180:
                ax_h_ph.axvline(bloom_day_ex, color="purple", lw=2,
                                label=f"t★ = day {bloom_day_ex}  ← bloom!")
            ax_h_ph.set_ylabel("Accumulated GDD", fontsize=9)
            ax_h_ph.set_xlabel("Day of year", fontsize=9)
            ax_h_ph.legend(fontsize=8, loc="upper left")
            ax_h_ph.set_title("Step 2 — H(t) crosses H★ → bloom event", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_ode_ph)
            plt.close(fig_ode_ph)

    # ── Section 2: The data ────────────────────────────────────────────────────
    with st.expander("📊 Section 2 — The data (synthetic, 1980–2019)", expanded=True):
        d1, d2 = st.columns([3, 2])

        with d1:
            st.markdown("**Spring temperature heatmap** (days 1–150, all years)")
            T_spring = T_daily_ph[:, :150]
            fig_hm = go.Figure(go.Heatmap(
                z=T_spring,
                x=list(range(1, 151)),
                y=[str(y) for y in years_ph],
                colorscale="RdBu_r",
                colorbar=dict(title="°C", len=0.6),
                zmin=-15, zmax=25,
            ))
            fig_hm.add_shape(type="line", x0=true_params_ph["t0"], x1=true_params_ph["t0"],
                             y0=0, y1=1, xref="x", yref="paper",
                             line=dict(color="black", width=1.5, dash="dot"))
            fig_hm.add_annotation(x=true_params_ph["t0"], y=1.02, xref="x", yref="paper",
                                  text="t₀", showarrow=False, font=dict(size=10))
            fig_hm.update_layout(xaxis_title="Day of year", yaxis_title="Year",
                                  height=380, margin=dict(t=30, b=40))
            st.plotly_chart(fig_hm, use_container_width=True)

        with d2:
            st.markdown("**Observed first-bloom DOY**")
            # Fit a trend line
            p_trend = np.polyfit(years_ph - 1980, doy_obs_ph, 1)
            trend_line = np.polyval(p_trend, years_ph - 1980)

            fig_doy = go.Figure()
            fig_doy.add_trace(go.Scatter(
                x=years_ph, y=doy_obs_ph, mode="markers",
                marker=dict(size=9, color=C_GREEN, line=dict(width=1, color="darkgreen")),
                name="Observed DOY",
            ))
            fig_doy.add_trace(go.Scatter(
                x=years_ph, y=trend_line, mode="lines",
                line=dict(color="black", width=1.5, dash="dash"),
                name=f"Trend ({p_trend[0]:+.2f} days/yr)",
            ))
            fig_doy.update_layout(xaxis_title="Year", yaxis_title="Bloom DOY",
                                   height=220, margin=dict(t=10, b=30),
                                   legend=dict(x=0.0, y=1.0))
            st.plotly_chart(fig_doy, use_container_width=True)

            st.markdown("**Histogram of bloom DOY**")
            fig_hist_doy = go.Figure(go.Histogram(
                x=doy_obs_ph, nbinsx=20,
                marker_color=C_GREEN, opacity=0.8,
            ))
            fig_hist_doy.update_layout(xaxis_title="Bloom DOY",
                                        yaxis_title="Count",
                                        height=180, margin=dict(t=10, b=30))
            st.plotly_chart(fig_hist_doy, use_container_width=True)

        # Warming trend caption
        st.caption(
            f"Synthetic data: warming rate +0.03 °C/yr. "
            f"True params: T_base={true_params_ph['T_base']} °C, "
            f"H★={true_params_ph['H_star']} GDD, "
            f"t₀=day {int(true_params_ph['t0'])}, σ={true_params_ph['sigma']} days."
        )

    # ── Section 3: PPL Metropolis ──────────────────────────────────────────────
    with st.expander("🔧 Section 3 — Fit with our PPL (Metropolis)", expanded=False):
        p3col, p3ctrl = st.columns([3, 1])

        with p3ctrl:
            st.markdown("**Sampler settings**")
            ph_n    = st.slider("n_samples", 2000, 15000, 6000, 1000, key="ph_n")
            ph_burn = st.slider("burn_in",    500,  5000, 2000,  500, key="ph_burn")
            st.markdown("**Per-parameter proposal σ**")
            ph_std_Tb = st.slider("T_base",  0.1, 2.0, 0.4, 0.1, key="ph_sTb")
            ph_std_Hs = st.slider("H_star",  1.0, 30.0, 8.0, 1.0, key="ph_sHs")
            ph_std_t0 = st.slider("t0",       0.5, 8.0, 2.5, 0.5, key="ph_st0")
            ph_std_sg = st.slider("sigma",    0.1, 2.0, 0.3, 0.1, key="ph_ssg")
            run_ph_ppl = st.button("Run PPL Metropolis",
                                   type="primary", key="ph_run_ppl")

        with p3col:
            st.markdown("""
### Building the likelihood

The ODE integration step is just a Python function that we wrap in our custom
`log_prob`:

```python
def make_log_prob(doy_obs, T_daily):
    def log_prob(state):
        T_base = state["T_base"];  H_star = state["H_star"]
        t0     = state["t0"];      sigma  = state["sigma"]
        if sigma <= 0 or H_star <= 0 or t0 < 1 or t0 > 120:
            return -np.inf
        # Priors
        lp  = sp.norm.logpdf(T_base, 2, 3)
        lp += sp.halfnorm.logpdf(H_star, scale=200)
        lp += sp.uniform.logpdf(t0, 1, 119)
        lp += sp.halfnorm.logpdf(sigma, scale=10)
        # Likelihood — ODE gives one predicted DOY per year
        doy_pred = integrate_gdd(T_daily, T_base, H_star, int(round(t0)))
        lp += np.sum(sp.norm.logpdf(doy_obs, doy_pred, sigma))
        return lp
    return log_prob

# Per-parameter proposal widths (parameters live on very different scales)
sampler = DirectMCMC(
    log_prob_fn   = make_log_prob(doy_obs, T_daily),
    initial_state = {"T_base": 2.0, "H_star": 200.0,
                     "t0": 60.0, "sigma": 5.0},
    proposal_std  = {"T_base": 0.4, "H_star": 8.0,
                     "t0": 2.5, "sigma": 0.3},
)
chain = sampler.sample(n_samples=6000, burn_in=2000)
```
""")

            if run_ph_ppl:
                log_prob_ph = make_log_prob_pheno(doy_obs_ph, T_daily_ph)
                prop_widths = {
                    "T_base": float(ph_std_Tb),
                    "H_star": float(ph_std_Hs),
                    "t0":     float(ph_std_t0),
                    "sigma":  float(ph_std_sg),
                }
                init_ph = {"T_base": 2.0, "H_star": 200.0, "t0": 60.0, "sigma": 5.0}
                dc_ph = _ppl.DirectMCMC(log_prob_ph, init_ph, prop_widths)
                with st.spinner(f"Running Metropolis ({int(ph_n)} samples)…"):
                    chain_ph = dc_ph.sample(int(ph_n), int(ph_burn))
                st.session_state["pheno_ppl_res"] = dict(
                    chain=chain_ph, rate=dc_ph.acceptance_rate)

        res_ph = st.session_state["pheno_ppl_res"]
        if res_ph:
            ch_ph = res_ph["chain"]
            st.metric("Acceptance rate", f"{res_ph['rate']:.1%}")

            ph_keys   = ["T_base", "H_star", "t0", "sigma"]
            ph_chains = {k: np.array([s[k] for s in ch_ph]) for k in ph_keys}
            ph_colors = [C_BLUE, C_RED, C_ORANGE, C_GREEN]

            # Trace + histogram grid
            fig_ph_diag, axes_ph = plt.subplots(2, 4, figsize=(14, 5))
            fig_ph_diag.suptitle("PPL Metropolis — Phenology Posterior",
                                  fontsize=12, fontweight="bold")
            for idx, k in enumerate(ph_keys):
                ch = ph_chains[k]
                col = ph_colors[idx]
                true_v = true_params_ph[k]

                axes_ph[0, idx].plot(ch, lw=0.4, color=col, alpha=0.8)
                axes_ph[0, idx].axhline(true_v, color="red", lw=1.2,
                                         ls="--", label="true")
                axes_ph[0, idx].set_title(f"{k}  (true={true_v})", fontsize=9)
                axes_ph[0, idx].legend(fontsize=7)

                axes_ph[1, idx].hist(ch, bins=50, color=col, alpha=0.7,
                                      density=True)
                axes_ph[1, idx].axvline(true_v, color="red", lw=1.5, ls="--")
                axes_ph[1, idx].axvline(ch.mean(), color="black", lw=1.5,
                                         label=f"mean={ch.mean():.2f}")
                axes_ph[1, idx].set_title(f"{k} posterior", fontsize=9)
                axes_ph[1, idx].legend(fontsize=7)

            plt.tight_layout()
            st.pyplot(fig_ph_diag)
            plt.close(fig_ph_diag)

            # Posterior predictive
            np.random.seed(0)
            idx_pp = np.random.choice(len(ch_ph), min(300, len(ch_ph)), replace=False)
            doy_pp = np.array([
                integrate_gdd_fast(T_daily_ph,
                                   ch_ph[i]["T_base"], ch_ph[i]["H_star"],
                                   int(round(ch_ph[i]["t0"])))
                for i in idx_pp
            ])
            doy_lo = np.percentile(doy_pp, 5, axis=0)
            doy_hi = np.percentile(doy_pp, 95, axis=0)
            doy_md = np.percentile(doy_pp, 50, axis=0)

            fig_ppc_ph = go.Figure()
            fig_ppc_ph.add_trace(go.Scatter(
                x=np.concatenate([years_ph, years_ph[::-1]]),
                y=np.concatenate([doy_hi, doy_lo[::-1]]),
                fill="toself", fillcolor="rgba(99,110,250,0.2)",
                line=dict(color="rgba(0,0,0,0)"), name="90% CI"))
            fig_ppc_ph.add_trace(go.Scatter(
                x=years_ph, y=doy_md, mode="lines",
                line=dict(color=C_BLUE, width=2), name="Posterior median"))
            fig_ppc_ph.add_trace(go.Scatter(
                x=years_ph, y=doy_obs_ph, mode="markers",
                marker=dict(size=9, color="black", symbol="x"), name="Observed"))
            fig_ppc_ph.update_layout(
                title="Posterior predictive retrodiction — PPL Metropolis",
                xaxis_title="Year", yaxis_title="Bloom DOY",
                height=320, margin=dict(t=50, b=20))
            st.plotly_chart(fig_ppc_ph, use_container_width=True)

        else:
            st.info("Configure the sampler and click **Run PPL Metropolis**.")

    # ── Section 4: PyMC NUTS ───────────────────────────────────────────────────
    with st.expander("⚡ Section 4 — Fit with PyMC (NUTS, soft threshold)", expanded=False):
        p4col, p4ctrl = st.columns([3, 1])

        with p4ctrl:
            st.markdown("**Sampler settings**")
            pymc_draws = st.slider("Draws per chain", 500, 2000, 1000, 250,
                                    key="ph_pymc_draws")
            pymc_tune  = st.slider("Tuning steps",    500, 2000, 1000, 250,
                                    key="ph_pymc_tune")
            run_ph_pymc = st.button("Run PyMC NUTS",
                                    type="primary", key="ph_run_pymc")

        with p4col:
            st.markdown("""
### The soft-threshold trick

NUTS needs **gradients**, but "find the first crossing" is a hard
step function that has zero gradient almost everywhere. We replace it
with a **differentiable soft approximation**:

1. **Soft t₀ mask** — instead of zeroing days before t₀, weight them
   with a sigmoid: days near t₀ contribute partially.
2. **Soft threshold crossing** — instead of an exact step, model the
   probability of having crossed H★ by day d as a sigmoid.
3. **Expected DOY** — sum d × P(bloom on day d) to get a smooth scalar.

```python
def predict_bloom_soft(T_base, H_star, t0, T_pt):
    # Soft t0 mask
    days   = pt.arange(365, dtype="float64")
    mask   = pt.sigmoid(days[None, :] - t0)        # (1, 365)
    heat   = pt.maximum(T_pt - T_base, 0.0) * mask # (N, 365)
    H_cum  = pt.cumsum(heat, axis=1)                # (N, 365)

    # Soft threshold: probability of having bloomed by day d
    p_crossed = pt.sigmoid(0.1 * (H_cum - H_star))

    # Probability of blooming on exactly day d
    p_prev  = pt.concatenate(
        [pt.zeros_like(p_crossed[:, :1]), p_crossed[:, :-1]], axis=1)
    p_daily = p_crossed - p_prev  # (N, 365)

    # Expected bloom DOY
    days_1 = pt.arange(1, 366, dtype="float64")[None, :]
    return pt.sum(p_daily * days_1, axis=1)         # (N,)
```

The model and likelihood are then standard PyMC:
```python
with pm.Model():
    T_base = pm.Normal("T_base", mu=2, sigma=3)
    H_star = pm.HalfNormal("H_star", sigma=200)
    t0     = pm.Uniform("t0", lower=1, upper=120)
    sigma  = pm.HalfNormal("sigma", sigma=10)
    doy_mu = predict_bloom_soft(T_base, H_star, t0, T_pt)
    pm.Normal("doy_obs", mu=doy_mu, sigma=sigma, observed=doy_obs)
    idata = pm.sample(draws=1000, tune=1000, target_accept=0.9)
```
""")

            if run_ph_pymc:
                try:
                    import pymc as pm
                    import pytensor.tensor as pt
                    import arviz as az
                    import logging
                    logging.getLogger("pymc").setLevel(logging.ERROR)
                    warnings.filterwarnings("ignore")

                    T_pt_np = T_daily_ph.astype("float64")
                    doy_np  = doy_obs_ph.astype("float64")

                    with pm.Model() as pm_pheno:
                        T_base_pm = pm.Normal("T_base", mu=2.0,  sigma=3.0)
                        H_star_pm = pm.HalfNormal("H_star",       sigma=200.0)
                        t0_pm     = pm.Uniform("t0",   lower=1.0, upper=120.0)
                        sigma_pm  = pm.HalfNormal("sigma",         sigma=10.0)

                        T_pt = pt.as_tensor_variable(T_pt_np)
                        days_pt = pt.arange(365, dtype="float64")
                        mask    = pt.sigmoid(days_pt[None, :] - t0_pm)
                        heat    = pt.maximum(T_pt - T_base_pm, 0.0) * mask
                        H_cum   = pt.cumsum(heat, axis=1)
                        p_crossed = pt.sigmoid(0.1 * (H_cum - H_star_pm))
                        p_prev  = pt.concatenate(
                            [pt.zeros_like(p_crossed[:, :1]), p_crossed[:, :-1]], axis=1)
                        p_daily = p_crossed - p_prev
                        days_1  = pt.arange(1, 366, dtype="float64")[None, :]
                        doy_mu  = pt.sum(p_daily * days_1, axis=1)

                        pm.Normal("doy_obs", mu=doy_mu, sigma=sigma_pm,
                                  observed=doy_np)

                        with st.spinner(
                            f"Running PyMC NUTS ({int(pymc_draws)} draws, "
                            f"{int(pymc_tune)} tune, 2 chains)…"
                        ):
                            idata_ph = pm.sample(
                                int(pymc_draws), tune=int(pymc_tune),
                                target_accept=0.9, chains=2, cores=1,
                                progressbar=False,
                                return_inferencedata=True,
                            )

                    st.session_state["pheno_pymc_res"] = dict(idata=idata_ph)

                except Exception as exc:
                    st.error(f"PyMC error: {exc}")

        res_pm_ph = st.session_state["pheno_pymc_res"]
        if res_pm_ph:
            import arviz as az
            idata_ph2 = res_pm_ph["idata"]
            pm_keys   = ["T_base", "H_star", "t0", "sigma"]

            # ArviZ summary with true values alongside
            st.markdown("### ArviZ posterior summary")
            summ_df = az.summary(idata_ph2, var_names=pm_keys)
            # Append true values column
            summ_df.insert(0, "true", [true_params_ph[k] for k in pm_keys])
            st.dataframe(summ_df.style.format("{:.3f}"), use_container_width=True)

            # Posterior histograms
            fig_pm_hist, axes_pm = plt.subplots(1, 4, figsize=(14, 3.5))
            fig_pm_hist.suptitle("PyMC NUTS — Phenology Posterior",
                                  fontsize=11, fontweight="bold")
            for idx, k in enumerate(pm_keys):
                arr = idata_ph2.posterior[k].values.flatten()
                tv  = true_params_ph[k]
                axes_pm[idx].hist(arr, bins=50, color=ph_colors[idx],
                                   alpha=0.7, density=True)
                axes_pm[idx].axvline(tv, color="red", lw=1.5, ls="--",
                                      label=f"true={tv}")
                axes_pm[idx].axvline(arr.mean(), color="black", lw=1.5,
                                      label=f"mean={arr.mean():.2f}")
                axes_pm[idx].set_title(k, fontsize=10)
                axes_pm[idx].legend(fontsize=7)
            plt.tight_layout()
            st.pyplot(fig_pm_hist)
            plt.close(fig_pm_hist)

            # Posterior predictive retrodiction
            T_base_pm_s = idata_ph2.posterior["T_base"].values.flatten()
            H_star_pm_s = idata_ph2.posterior["H_star"].values.flatten()
            t0_pm_s     = idata_ph2.posterior["t0"].values.flatten()
            idx_rtr = np.random.choice(len(T_base_pm_s), 300, replace=False)
            doy_rtr = np.array([
                integrate_gdd_fast(T_daily_ph,
                                   T_base_pm_s[i], H_star_pm_s[i],
                                   int(round(t0_pm_s[i])))
                for i in idx_rtr
            ])
            rtr_lo = np.percentile(doy_rtr, 5,  axis=0)
            rtr_hi = np.percentile(doy_rtr, 95, axis=0)
            rtr_md = np.percentile(doy_rtr, 50, axis=0)

            fig_rtr = go.Figure()
            fig_rtr.add_trace(go.Scatter(
                x=np.concatenate([years_ph, years_ph[::-1]]),
                y=np.concatenate([rtr_hi, rtr_lo[::-1]]),
                fill="toself", fillcolor="rgba(239,85,59,0.2)",
                line=dict(color="rgba(0,0,0,0)"), name="90% CI"))
            fig_rtr.add_trace(go.Scatter(
                x=years_ph, y=rtr_md, mode="lines",
                line=dict(color=C_RED, width=2), name="Posterior median"))
            fig_rtr.add_trace(go.Scatter(
                x=years_ph, y=doy_obs_ph, mode="markers",
                marker=dict(size=9, color="black", symbol="x"), name="Observed"))
            fig_rtr.update_layout(
                title="Posterior predictive retrodiction — PyMC NUTS",
                xaxis_title="Year", yaxis_title="Bloom DOY",
                height=300, margin=dict(t=50, b=20))
            st.plotly_chart(fig_rtr, use_container_width=True)

            # Pair plot — T_base vs H_star
            st.markdown("### T_base vs H_star — posterior correlation")
            st.markdown("""
These two parameters trade off: if $T_{\\text{base}}$ is higher, each day
contributes less heat, so $H^\\star$ must be **lower** to bloom at the same time.
NUTS handles this negative correlation naturally via gradient information.
Metropolis, which proposes each parameter independently, struggles.
""")
            fig_pair = go.Figure()
            fig_pair.add_trace(go.Scatter(
                x=T_base_pm_s[::2], y=H_star_pm_s[::2],
                mode="markers",
                marker=dict(size=3, opacity=0.3, color=C_RED),
                name="Posterior samples"))
            fig_pair.add_trace(go.Scatter(
                x=[true_params_ph["T_base"]], y=[true_params_ph["H_star"]],
                mode="markers",
                marker=dict(size=16, color="black", symbol="cross"),
                name="True values"))
            fig_pair.update_layout(
                xaxis_title="T_base (°C)", yaxis_title="H★ (GDD)",
                title="Posterior correlation — negative trade-off between T_base and H★",
                height=340, margin=dict(t=60, b=30))
            st.plotly_chart(fig_pair, use_container_width=True)

        else:
            st.info("Click **Run PyMC NUTS** to fit the model.")

    # ── Section 5: Comparison ──────────────────────────────────────────────────
    with st.expander("⚖️ Section 5 — Compare both samplers", expanded=False):
        res_ppl_cmp  = st.session_state["pheno_ppl_res"]
        res_pymc_cmp = st.session_state["pheno_pymc_res"]

        if res_ppl_cmp and res_pymc_cmp:
            import arviz as az
            ph_keys_cmp = ["T_base", "H_star", "t0", "sigma"]

            st.markdown("### Posterior violin plots")
            ch_cmp   = res_ppl_cmp["chain"]
            idata_cmp = res_pymc_cmp["idata"]

            fig_viol_ph, axes_viol = plt.subplots(1, 4, figsize=(14, 5))
            fig_viol_ph.suptitle(
                "PPL Metropolis (blue) vs PyMC NUTS (red) — posterior marginals",
                fontsize=11, fontweight="bold")
            for idx, k in enumerate(ph_keys_cmp):
                ppl_arr  = np.array([s[k] for s in ch_cmp])
                nuts_arr = idata_cmp.posterior[k].values.flatten()
                tv       = true_params_ph[k]

                parts = axes_viol[idx].violinplot(
                    [ppl_arr, nuts_arr], positions=[0, 1],
                    showmedians=True, showextrema=False)
                parts["bodies"][0].set_facecolor(C_BLUE)
                parts["bodies"][0].set_alpha(0.55)
                parts["bodies"][1].set_facecolor(C_RED)
                parts["bodies"][1].set_alpha(0.55)
                axes_viol[idx].axhline(tv, color="black", lw=1.5,
                                        ls="--", label=f"true={tv}")
                axes_viol[idx].set_xticks([0, 1])
                axes_viol[idx].set_xticklabels(["PPL\nMetropolis", "PyMC\nNUTS"])
                axes_viol[idx].set_title(k, fontsize=10)
                axes_viol[idx].legend(fontsize=7)
            plt.tight_layout()
            st.pyplot(fig_viol_ph)
            plt.close(fig_viol_ph)

            st.markdown("### Numerical comparison")
            cmp_rows = {}
            for k in ph_keys_cmp:
                ppl_v  = np.array([s[k] for s in ch_cmp])
                nuts_v = idata_cmp.posterior[k].values.flatten()
                cmp_rows[k] = {
                    "True":       round(true_params_ph[k], 3),
                    "PPL mean":   round(float(ppl_v.mean()),  3),
                    "PPL std":    round(float(ppl_v.std()),   3),
                    "NUTS mean":  round(float(nuts_v.mean()), 3),
                    "NUTS std":   round(float(nuts_v.std()),  3),
                }
            st.dataframe(pd.DataFrame(cmp_rows).T, use_container_width=True)

            st.markdown("""
### What to take away

**The tradeoff at the heart of this case study:**

| | PPL Metropolis | PyMC NUTS |
|---|---|---|
| **Model** | Hard threshold — exact biology | Soft sigmoid — approximate |
| **Gradients** | Not needed | Required |
| **Handles T_base / H★ correlation** | Poorly — proposes them independently | Well — gradient encodes the trade-off |
| **Tuning effort** | Must set per-parameter σ by hand | Self-tunes during warmup |
| **Chain length needed** | Long (high autocorrelation near correlated ridge) | Short (near-independent draws) |

**Why Metropolis struggles here.**
$T_{\\text{base}}$ and $H^\\star$ sit on a narrow curved ridge in parameter
space (the negative correlation you saw in the pair plot). Metropolis
proposes each parameter independently with a fixed Gaussian, so most
proposals land off the ridge and are rejected. The sampler creeps slowly
along it, producing highly autocorrelated chains.

**Why NUTS handles it better.**
The gradient of the log-posterior points *along* the ridge, so NUTS
proposals follow the ridge naturally, exploring it efficiently without
manual tuning.

**The cost: soft vs. hard threshold.**
NUTS required replacing the biologically exact hard threshold with a
smooth sigmoid approximation. For this problem the approximation is
excellent, but in general you must decide whether the model fidelity
lost is acceptable — or invest in a bespoke gradient computation.
""")

        else:
            st.info("Run both **Section 3** (PPL) and **Section 4** (PyMC) first "
                    "to see the comparison.")

