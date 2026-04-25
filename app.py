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

# ── Session state init ────────────────────────────────────────────────────────

for _k in ("ov_step", "mh_step", "hmc_step", "ppl_step", "mh_res", "hmc_res",
           "ppl_samples", "glacier_ppl_res", "glacier_pymc_res"):
    if _k not in st.session_state:
        st.session_state[_k] = None if _k.endswith(("res", "samples")) else 0

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_ov, tab_mh, tab_hmc, tab_cmp, tab_ppl, tab_glacier = st.tabs([
    "Overview", "Metropolis-Hastings", "HMC & Leapfrog", "Comparison",
    "🔧 Build a PPL", "🏔 Glacier Case Study",
])

# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_ov:
    st.title("Markov Chain Monte Carlo — Interactive Explorer")

    OV_LABELS = [
        "The core problem",
        "Bayesian inference",
        "A worked example",
        "Where it's used",
        "The two algorithms",
    ]
    ov_step = step_nav("ov_step", OV_LABELS)

    # ── Step 0: The core problem ───────────────────────────────────────────────
    if ov_step == 0:
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.markdown("## The core problem")
            st.markdown("""
Many problems in statistics, physics, and machine learning reduce to computing **expectations** under a probability distribution π:
""")
            st.latex(r"\mathbb{E}_{\theta \sim \pi}[f(\theta)] = \int f(\theta)\,\pi(\theta)\,d\theta")
            st.markdown("""
This integral rarely has a closed form. And even numerically, it becomes **exponentially harder** as the number of dimensions grows — with 50 parameters and a 10-point grid per dimension you'd need 10⁵⁰ evaluations.

**Monte Carlo** sidesteps this with samples: draw θ₁, …, θ_N ~ π and use the average:
""")
            st.latex(r"\mathbb{E}[f(\theta)] \;\approx\; \frac{1}{N}\sum_{i=1}^{N} f(\theta_i)")
            st.markdown("""
The error is **O(1/√N)** — independent of dimension. The catch: how do you *draw* samples from π when π is complex and high-dimensional?

If you can write down π directly (e.g. a Gaussian), you can sample it analytically. But for most real models — Bayesian posteriors, Boltzmann distributions, latent variable models — you cannot. You only know π **up to a normalizing constant**:
""")
            st.latex(r"\pi(\theta) = \frac{\tilde\pi(\theta)}{Z}, \qquad Z = \int \tilde\pi(\theta)\,d\theta \text{ unknown}")
            st.markdown("""
**MCMC** is the answer: construct a Markov chain that is *easy to simulate* and whose stationary distribution is *exactly π*, without ever computing Z.
""")
            st.info("Click **Next ▶** to see how Bayesian inference creates this exact situation.")
        with col_r:
            # Illustrate the curse of dimensionality + Monte Carlo convergence
            rng_ov = np.random.default_rng(7)
            ns = [10, 50, 200, 1000, 5000]
            errs = []
            for n in ns:
                s = rng_ov.standard_normal((200, n))
                mc_est = np.mean(np.abs(s), axis=1)   # E[|X|] for N(0,1) = sqrt(2/pi)
                errs.append(float(np.std(mc_est)))
            true_val = np.sqrt(2.0 / np.pi)
            xs_cod = np.linspace(5, 6000, 300)

            fig_cod = go.Figure()
            fig_cod.add_trace(go.Scatter(
                x=xs_cod, y=1.0 / np.sqrt(xs_cod),
                mode="lines", line=dict(color=C_GREEN, width=2.5, dash="dash"),
                name="O(1/√N) theory",
            ))
            fig_cod.add_trace(go.Scatter(
                x=ns, y=errs, mode="markers+lines",
                marker=dict(size=10, color=C_BLUE),
                line=dict(color=C_BLUE, width=2),
                name="MC error (empirical)",
            ))
            fig_cod.update_layout(
                title="Monte Carlo error decays as O(1/√N) — independent of dimension",
                xaxis_title="Number of samples N",
                yaxis_title="Std. error of estimate",
                height=360, margin=dict(t=70, b=30),
                legend=dict(x=0.45, y=0.95),
            )
            st.plotly_chart(fig_cod, use_container_width=True)

            st.info("""
**Reading the diagnostics** (used throughout the app)

| | Good sign |
|---|---|
| Trace | Noisy, no trend |
| Histogram | Matches green density |
| ACF | Decays quickly to 0 |
| Acceptance | MH: 20–50 % · HMC: 60–90 % |
""")

    # ── Step 1: Bayesian inference ─────────────────────────────────────────────
    elif ov_step == 1:
        col_l, col_m, col_r = st.columns(3)

        with col_l:
            st.markdown("#### Frequentist view")
            st.markdown("""
Parameters **θ** are fixed but unknown constants. Data **y** is the random quantity.

Inference answers: *"If θ equalled some null value, how surprising would this data be?"*

Tools: p-values, confidence intervals, maximum-likelihood estimates (MLEs).

A **95% confidence interval** means: if we repeated the experiment many times, 95% of the constructed intervals would contain the true θ. It says nothing about the probability that *this particular* interval contains θ.
""")

        with col_m:
            st.markdown("#### Bayesian view")
            st.markdown("""
Parameters **θ** are treated as random variables. Inference answers: *"Given the data I observed, what should I believe about θ?"*

Formalised by **Bayes' theorem**:
""")
            st.latex(r"\underbrace{p(\theta \mid y)}_{\text{posterior}} \;\propto\; \underbrace{p(y \mid \theta)}_{\text{likelihood}} \;\cdot\; \underbrace{p(\theta)}_{\text{prior}}")
            st.markdown("""
A **95% credible interval** directly means: given the data, there is a 95% probability that θ lies in this interval.

The full posterior is useful for predictions, decisions, and propagating uncertainty downstream.
""")

        with col_r:
            st.markdown("#### Where MCMC enters")
            st.markdown("""
The exact posterior requires the normalizing constant:
""")
            st.latex(r"p(\theta \mid y) = \frac{p(y \mid \theta)\,p(\theta)}{\underbrace{\int p(y \mid \theta)\,p(\theta)\,d\theta}_{Z \;=\; \text{intractable}}}")
            st.markdown("""
For most real models Z has **no closed form** — and numerical quadrature fails once the parameter space has more than ~4 dimensions.

**MCMC only needs the numerator** — the unnormalized product p(y|θ)·p(θ) — evaluated pointwise. The chain is designed to explore θ-space proportionally to this product, automatically giving samples from the posterior.

→ Step 3 shows a concrete model where this is necessary.
""")

    # ── Step 2: Worked example — Bayesian logistic regression ─────────────────
    elif ov_step == 2:
        st.markdown("## A concrete example: Bayesian logistic regression")
        st.markdown("""
Here is the simplest model where you **provably need MCMC** (or a similar method).

**Setup**: 12 students studied for between 0 and 10 hours. Did they pass?
We model the pass probability with a logistic curve:
""")
        st.latex(r"P(\text{pass} \mid x, \beta_0, \beta_1) = \sigma(\beta_0 + \beta_1 x), \qquad \sigma(z) = \frac{1}{1+e^{-z}}")
        st.markdown("""
We put a gentle prior on the coefficients: β₀, β₁ ~ N(0, 2²).

The **posterior** is:
""")
        st.latex(r"p(\beta_0, \beta_1 \mid \text{data}) \;\propto\; \underbrace{\prod_{i=1}^{n} \sigma(\beta_0+\beta_1 x_i)^{y_i}(1-\sigma(\beta_0+\beta_1 x_i))^{1-y_i}}_{\text{likelihood}} \cdot \underbrace{e^{-(\beta_0^2+\beta_1^2)/8}}_{\text{prior}}")

        # Fixed dataset
        x_data = np.array([0.5, 1.0, 2.0, 2.5, 3.5, 4.0, 5.0, 6.0, 7.0, 7.5, 9.0, 10.0])
        y_data = np.array([0,   0,   0,   1,   0,   1,   1,   0,   1,   1,   1,   1  ])

        def log_posterior_logreg(b0, b1):
            z = b0 + b1 * x_data
            z = np.clip(z, -30, 30)
            log_lik = np.sum(y_data * (-np.log1p(np.exp(-z)))
                             + (1 - y_data) * (-np.log1p(np.exp(z))))
            log_prior = -0.5 * (b0**2 + b1**2) / 4.0
            return log_lik + log_prior

        b0_grid = np.linspace(-4, 4, 180)
        b1_grid = np.linspace(-0.5, 2.5, 180)
        B0, B1 = np.meshgrid(b0_grid, b1_grid)
        log_post = np.array([[log_posterior_logreg(b0, b1)
                               for b0 in b0_grid] for b1 in b1_grid])
        log_post -= log_post.max()

        col_l, col_r = st.columns([1, 1])
        with col_l:
            fig_data = go.Figure()
            x_curve = np.linspace(0, 10.5, 200)
            # MLE (rough) for the curve — use the posterior mode region
            best_idx = np.unravel_index(log_post.argmax(), log_post.shape)
            b0_mode = float(b0_grid[best_idx[1]])
            b1_mode = float(b1_grid[best_idx[0]])
            p_curve = 1.0 / (1.0 + np.exp(-(b0_mode + b1_mode * x_curve)))

            fig_data.add_trace(go.Scatter(
                x=x_data[y_data == 0], y=y_data[y_data == 0],
                mode="markers", marker=dict(size=14, color=C_RED, symbol="circle"),
                name="Fail (y=0)",
            ))
            fig_data.add_trace(go.Scatter(
                x=x_data[y_data == 1], y=y_data[y_data == 1],
                mode="markers", marker=dict(size=14, color=C_GREEN, symbol="circle"),
                name="Pass (y=1)",
            ))
            fig_data.add_trace(go.Scatter(
                x=x_curve, y=p_curve, mode="lines",
                line=dict(color=C_BLUE, width=2.5),
                name="Posterior-mode fit",
            ))
            fig_data.update_layout(
                title="The data: study hours vs. pass/fail",
                xaxis_title="Hours studied (x)",
                yaxis=dict(title="P(pass)", range=[-0.08, 1.08],
                           tickvals=[0, 0.5, 1], ticktext=["0 (Fail)", "0.5", "1 (Pass)"]),
                height=340, margin=dict(t=60, b=30),
                legend=dict(x=0.02, y=0.95),
            )
            st.plotly_chart(fig_data, use_container_width=True)

            st.markdown("""
**Why can't we just integrate?**

The posterior over (β₀, β₁) involves a product of logistic functions. There is no formula for the integral — not even in 2D. The normalizing constant Z = ∫∫ p(data|β)·p(β) dβ₀ dβ₁ must be computed numerically, and in higher-dimensional models this becomes completely infeasible.

MCMC draws samples from this posterior **without computing Z**.
""")

        with col_r:
            fig_post = go.Figure()
            fig_post.add_trace(go.Contour(
                x=b0_grid, y=b1_grid, z=np.exp(log_post),
                colorscale="Blues",
                contours=dict(showlabels=False),
                colorbar=dict(title="unnorm. posterior", len=0.6),
                name="Posterior",
            ))
            fig_post.update_layout(
                title="Unnormalized posterior p(β₀, β₁ | data)",
                xaxis_title="β₀  (intercept)",
                yaxis_title="β₁  (slope)",
                height=340, margin=dict(t=60, b=30),
            )
            st.plotly_chart(fig_post, use_container_width=True)

            st.markdown("""
**What MCMC gives you**

Once you have samples (β₀⁽ⁱ⁾, β₁⁽ⁱ⁾) from the posterior you can:

- Compute a **credible interval** for the probability of passing at 5 hours:
  just evaluate σ(β₀⁽ⁱ⁾ + 5β₁⁽ⁱ⁾) for each sample and report the 2.5–97.5 percentile range
- Answer *"given this dataset, is there a >90% chance a student who studies 8 hours passes?"*
- Propagate uncertainty into any downstream decision

None of this requires knowing Z.
""")

    # ── Step 3: Where it's used ────────────────────────────────────────────────
    elif ov_step == 3:
        st.markdown("## Where is this actually used?")
        ex1, ex2, ex3 = st.columns(3)

        with ex1:
            st.markdown("#### Clinical trials")
            st.markdown("""
A drug trial records whether each patient responded to treatment. We want to estimate the true response rate θ and quantify uncertainty.

**Simple case** (single rate): prior Beta(1,1), posterior Beta(1+successes, 1+failures) — has a closed form, no MCMC needed.

**Realistic case**: adjust for covariates (age, weight, dose) via logistic regression. The posterior over all regression coefficients has no closed form → MCMC.

The posterior gives a **full distribution over plausible effect sizes**, not just a point estimate, which is more useful for clinical decisions.
""")

        with ex2:
            st.markdown("#### Hierarchical models")
            st.markdown("""
Many datasets have **grouped structure**: students within schools, patients within hospitals. A hierarchical model pools information while estimating group-specific effects.
""")
            st.latex(r"""\mu_j \sim \mathcal{N}(\mu, \sigma^2_\mu), \quad
y_{ij} \sim \mathcal{N}(\mu_j, \sigma^2_e)""")
            st.markdown("""
The posterior over all school means {μⱼ}, global mean μ, and variance components has no closed form — the coupling between parameters makes direct sampling impossible even with Gaussian likelihoods.

This is one of the most common uses of MCMC in social science, educational research, and medicine.
""")

        with ex3:
            st.markdown("#### Physics & statistical mechanics")
            st.markdown("""
Equilibrium distributions follow the **Boltzmann distribution**:
""")
            st.latex(r"p(\mathbf{x}) \propto \exp\!\left(-\frac{E(\mathbf{x})}{k_B T}\right)")
            st.markdown("""
where **x** is the system's configuration (e.g. atomic positions) and E is its energy.

Computing macroscopic properties (pressure, phase transitions) requires averaging over this. With 10²³ particles direct integration is hopeless.

The Metropolis algorithm was invented for exactly this (Metropolis et al., 1953). HMC grew from molecular dynamics. Both are still heavily used in computational chemistry and lattice QCD today.
""")

    # ── Step 4: The two algorithms ─────────────────────────────────────────────
    elif ov_step == 4:
        st.markdown("## The two algorithms in this app")
        col_l, col_r = st.columns([1, 1])

        with col_l:
            st.markdown("#### Metropolis-Hastings")
            st.markdown("""
The simplest MCMC algorithm. From the current position q:

1. **Propose** q* ~ N(q, σ²)
2. **Accept** with probability min(1, π(q*)/π(q))
3. Move to q* if accepted, stay at q otherwise

The chain drifts toward high-density regions because uphill moves are always accepted; downhill moves are only sometimes accepted.

**Drawback**: it moves by a random walk, so to travel a distance d it takes O(d²) steps. For complex posteriors this is very slow.
""")
            st.info("→ Metropolis-Hastings tab walks through this step-by-step.")

        with col_r:
            st.markdown("#### Hamiltonian Monte Carlo")
            st.markdown("""
A smarter algorithm that uses the **gradient** of log π to make long, directed proposals:

1. Draw auxiliary momentum **p ~ N(0,1)**
2. Run L **leapfrog steps** (a symplectic ODE integrator) to propose (q*, p*)
3. Accept with min(1, exp(−ΔH)) — usually near 100%

The leapfrog integrator conserves the Hamiltonian H = U(q) + K(p), so proposals travel far along the density surface without random-walk diffusion.

**Why leapfrog?** It's *symplectic* — it keeps energy error bounded for all time. Euler's method drifts and causes proposals to be rejected.
""")
            st.info("→ HMC & Leapfrog tab walks through this step-by-step.")

        st.divider()
        st.markdown("""
| | Metropolis-Hastings | Hamiltonian Monte Carlo |
|---|---|---|
| **Proposal** | Random Gaussian jump | Gradient-guided leap via Hamiltonian dynamics |
| **Mixing speed** | Slow — random walk scaling O(d²) | Fast — can traverse the space in O(d) steps |
| **What it needs** | log π(q) only | log π(q) **and** ∇log π(q) |
| **Key tuning** | Proposal width σ | Step size ε, leapfrog steps L |
| **Acceptance rate** | Target ~23–50 % | Target ~60–90 % |
| **Historical origin** | Metropolis et al. (1953) for nuclear physics | Duane et al. (1987) for lattice QCD |

Use the tabs above to step through each algorithm interactively, then compare them head-to-head in the **Comparison** tab.
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
# GLACIER CASE STUDY TAB
# ─────────────────────────────────────────────────────────────────────────────
with tab_glacier:
    st.header("Case Study — Sperry Glacier Mass Balance")
    st.markdown("""
Glacier National Park's **Sperry Glacier** has been losing mass steadily since systematic
monitoring began in 2005. We'll fit a simple physical ODE model to 21 years of real USGS
mass-balance data using **(a) our PPL** and **(b) PyMC with NUTS**, then compare both
posteriors side-by-side.
""")

    # ── Load data ──────────────────────────────────────────────────────────────
    DATA_PATH = os.path.join(os.path.dirname(__file__), "Sperry_annual_mass_balance.csv")

    @st.cache_data(show_spinner=False)
    def load_sperry():
        df = pd.read_csv(DATA_PATH)
        years   = df["Year"].values.astype(int)
        Ba      = df["Ba"].values.astype(float)   # annual balance (m w.e.) — OBSERVED
        Bw      = df["Bw"].values.astype(float)   # winter balance  → P forcing
        Bs      = df["Bs"].values.astype(float)   # summer balance  → T forcing
        P_acc   = Bw                               # accumulation (m w.e./yr)
        T_melt  = -Bs                              # melt index (m w.e./yr, positive)
        return years, Ba, Bw, Bs, P_acc, T_melt

    years_sp, Ba_sp, Bw_sp, Bs_sp, P_acc_sp, T_melt_sp = load_sperry()
    n_years_sp = len(years_sp)

    # ── ODE simulation ─────────────────────────────────────────────────────────
    def glacier_ode(mu, c, T0, tau, V0, T_melt_arr, P_acc_arr):
        """
        Forward Euler integration of:
          dV/dt = c·P(t) − μ·max(T_melt(t) − T₀, 0) − V(t)/τ
        B_t = dV/dt  (specific mass balance in m w.e./yr)
        """
        n  = len(T_melt_arr)
        V  = float(V0)
        Bsim = np.zeros(n)
        for t in range(n):
            melt_t = float(mu) * max(float(T_melt_arr[t]) - float(T0), 0.0)
            accum_t = float(c) * float(P_acc_arr[t])
            dV_t   = accum_t - melt_t - V / float(tau)
            Bsim[t] = dV_t
            V += dV_t
        return Bsim

    # ── Section 1: The physical model ──────────────────────────────────────────
    with st.expander("📐 Section 1 — The physical model", expanded=True):
        eL, eR = st.columns([1, 1])
        with eL:
            st.markdown("### The ODE")
            st.latex(r"\frac{dV}{dt} = \underbrace{c \cdot P(t)}_{\text{accumulation}} - \underbrace{\mu \cdot \max(T(t) - T_0,\, 0)}_{\text{melt}} - \underbrace{\frac{V(t)}{\tau}}_{\text{calving/outflow}}")
            st.markdown("""
**Observed variable:** Annual specific mass balance  B_t = dV/dt (m w.e./yr)

**Known forcings** (derived from seasonal observations):
- P(t) = winter mass balance Bw(t) — accumulation proxy (m w.e./yr)
- T(t) = −summer mass balance |Bs(t)| — melt index (m w.e./yr)

We integrate with forward Euler (dt = 1 yr), which is sufficient for this annual model.
""")
        with eR:
            st.markdown("### Parameters to infer")
            st.markdown("""
| Parameter | Description | Prior |
|---|---|---|
| **μ** | Melt sensitivity (fraction of melt index that drives volume loss) | HalfNormal(1) |
| **c** | Accumulation efficiency | HalfNormal(1) |
| **T₀** | Melt threshold (m w.e./yr) | Normal(2, 1.5) |
| **τ** | Volume response time (years) | Exponential(lam=1/20) |
| **V₀** | Initial volume (m w.e. equivalent) | Normal(10, 5) |
| **σ** | Observation noise (m w.e./yr) | HalfNormal(0.5) |

We have **21 data points** (2005–2025) and **6 parameters** — a realistic,
moderately constrained inference problem.  The parameters are correlated in the
posterior, making MCMC the natural tool.
""")

    # ── Section 2: The data ────────────────────────────────────────────────────
    with st.expander("📊 Section 2 — The data (Sperry Glacier, USGS)", expanded=True):
        fig_data_sp = make_subplots(rows=3, cols=1,
            subplot_titles=["Accumulation forcing P(t) = Bw (m w.e./yr)",
                            "Melt index T(t) = −Bs (m w.e./yr)",
                            "Annual mass balance Ba — OBSERVED (m w.e./yr)"],
            vertical_spacing=0.12)
        fig_data_sp.add_trace(go.Bar(x=years_sp, y=P_acc_sp,
                                     marker_color="steelblue", name="P(t) = Bw"), 1, 1)
        fig_data_sp.add_trace(go.Bar(x=years_sp, y=T_melt_sp,
                                     marker_color="tomato", name="T(t) = −Bs"), 2, 1)
        fig_data_sp.add_trace(go.Bar(x=years_sp, y=Ba_sp,
                                     marker_color=[C_GREEN if v >= 0 else C_RED for v in Ba_sp],
                                     name="Ba (observed)"), 3, 1)
        fig_data_sp.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        fig_data_sp.add_hline(y=float(Ba_sp.mean()), line_dash="dot", line_color="black",
                               annotation_text=f"mean={Ba_sp.mean():.2f}",
                               annotation_position="bottom right", row=3, col=1)
        fig_data_sp.update_layout(height=560, margin=dict(t=60, b=20),
                                   showlegend=False,
                                   title="Sperry Glacier, Glacier National Park MT — USGS data 2005–2025")
        st.plotly_chart(fig_data_sp, use_container_width=True)
        st.caption("Data: USGS Benchmark Glacier Program, doi:10.5066/F7HD7SRF, calibrated v10.0 (2026).")

    # ── Section 3: PPL inference ───────────────────────────────────────────────
    with st.expander("🔧 Section 3 — Fit with our PPL (Metropolis)", expanded=False):
        p3L, p3R = st.columns([1, 2])
        with p3L:
            st.markdown("### Metropolis on the glacier posterior")
            st.code("""\
# Custom likelihood wrapping the ODE
class GlacierLik:
    def __init__(self, T_melt, P_acc, B_obs, vars):
        self.T_melt = T_melt; self.P_acc = P_acc
        self.B_obs  = B_obs;  self.vars  = vars
    def log_prob(self, x):
        v = self.vars
        B_sim = glacier_ode(
            v['mu'].value, v['c'].value, v['T0'].value,
            v['tau'].value, v['V0'].value,
            self.T_melt, self.P_acc)
        sigma = v['sigma'].value
        if sigma <= 0: return -np.inf
        return float(np.sum(
            sp.norm.logpdf(self.B_obs, B_sim, sigma)))

# Build the model
model = Model()
mu_v  = model.add_variable('mu',    HalfNormal(1))
c_v   = model.add_variable('c',     HalfNormal(1))
T0_v  = model.add_variable('T0',    Normal(2, 1.5))
tau_v = model.add_variable('tau',   Exponential(1/20))
V0_v  = model.add_variable('V0',    Normal(10, 5))
sg_v  = model.add_variable('sigma', HalfNormal(0.5))
obs_v = model.add_variable('obs', GlacierLik(...),
        parents=[mu_v,c_v,T0_v,tau_v,V0_v,sg_v],
        observed=True)
""", language="python")
            ppl_gl_n    = st.slider("n_samples", 1000, 8000, 4000, 1000, key="gl_n")
            ppl_gl_burn = st.slider("burn_in",    500, 3000, 2000,  500, key="gl_burn")
            ppl_gl_std  = st.slider("proposal_std", 0.01, 0.20, 0.05, 0.01, key="gl_std")
            run_ppl_gl  = st.button("Run PPL Metropolis", type="primary", key="gl_ppl_run")

        with p3R:
            if run_ppl_gl:
                import scipy.stats as _sp2

                class _GlacierLik:
                    def __init__(self, T_melt, P_acc, B_obs, vdict):
                        self.T_melt = T_melt; self.P_acc = P_acc
                        self.B_obs = B_obs;   self.vdict = vdict
                    def log_prob(self, x):
                        v = self.vdict
                        try:
                            mu_val  = v["mu"].value
                            c_val   = v["c"].value
                            T0_val  = v["T0"].value
                            tau_val = v["tau"].value
                            V0_val  = v["V0"].value
                            sg_val  = v["sigma"].value
                            if sg_val is None or sg_val <= 0: return -np.inf
                            if tau_val is None or tau_val <= 0: return -np.inf
                            B_sim = glacier_ode(mu_val, c_val, T0_val, tau_val, V0_val,
                                                self.T_melt, self.P_acc)
                            return float(np.sum(_sp2.norm.logpdf(self.B_obs, B_sim, sg_val)))
                        except Exception:
                            return -np.inf

                m_gl   = _ppl.Model()
                mu_gl  = m_gl.add_variable("mu",    _ppl.HalfNormal(1.0))
                c_gl   = m_gl.add_variable("c",     _ppl.HalfNormal(1.0))
                T0_gl  = m_gl.add_variable("T0",    _ppl.Normal(2.0, 1.5))
                tau_gl = m_gl.add_variable("tau",   _ppl.Exponential(rate=1.0/20.0))
                V0_gl  = m_gl.add_variable("V0",    _ppl.Normal(10.0, 5.0))
                sg_gl  = m_gl.add_variable("sigma", _ppl.HalfNormal(0.5))
                vdict  = {"mu": mu_gl, "c": c_gl, "T0": T0_gl,
                          "tau": tau_gl, "V0": V0_gl, "sigma": sg_gl}
                m_gl.add_variable("obs", _GlacierLik(T_melt_sp, P_acc_sp, Ba_sp, vdict),
                                   parents=list(vdict.values()), observed=True, observed_data=Ba_sp)

                init_gl = {"mu": 0.80, "c": 0.58, "T0": 2.0, "tau": 20.0, "V0": 10.0, "sigma": 0.4}
                mc_gl = _ppl.MCMC(m_gl, initial_state=init_gl, proposal_std=float(ppl_gl_std))
                with st.spinner(f"Running Metropolis on glacier model ({int(ppl_gl_n)} samples)…"):
                    samps_gl = mc_gl.sample("metropolis", int(ppl_gl_n), int(ppl_gl_burn))

                st.session_state["glacier_ppl_res"] = dict(
                    samples=samps_gl, rate=mc_gl.acceptance_rate)

            res_gl = st.session_state["glacier_ppl_res"]
            if res_gl:
                st.metric("Acceptance rate", f"{res_gl['rate']:.1%}")
                sgl = res_gl["samples"]
                gl_params = ["mu", "c", "T0", "tau", "V0", "sigma"]
                gl_chains = {k: np.array([s[k] for s in sgl]) for k in gl_params}

                # Posterior means summary
                st.markdown("**Posterior means:**")
                pm_cols = st.columns(len(gl_params))
                for i, k in enumerate(gl_params):
                    pm_cols[i].metric(k, f"{gl_chains[k].mean():.3f}",
                                       delta=f"±{gl_chains[k].std():.3f}")

                # Trace + histogram for each
                fig_gl_diag, axes_gl = plt.subplots(2, 3, figsize=(13, 6))
                fig_gl_diag.suptitle("PPL Metropolis — Glacier Posterior", fontsize=12, fontweight="bold")
                colors_gl = [C_BLUE, C_ORANGE, C_GREEN, C_RED, "#9467bd", "#8c564b"]
                for idx, k in enumerate(gl_params):
                    r, c2 = divmod(idx, 3)
                    axes_gl[r, c2].plot(gl_chains[k], lw=0.5, color=colors_gl[idx], alpha=0.8)
                    axes_gl[r, c2].set_title(f"{k}  mean={gl_chains[k].mean():.3f}", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig_gl_diag)
                plt.close(fig_gl_diag)

                # Posterior predictive retrodiction
                np.random.seed(0)
                B_post_pred = np.array([
                    glacier_ode(s["mu"], s["c"], s["T0"], s["tau"], s["V0"], T_melt_sp, P_acc_sp)
                    for s in sgl[::max(1, len(sgl)//200)]
                ])
                Bpp_lo = np.percentile(B_post_pred, 5, axis=0)
                Bpp_hi = np.percentile(B_post_pred, 95, axis=0)
                Bpp_md = np.percentile(B_post_pred, 50, axis=0)

                fig_ret = go.Figure()
                fig_ret.add_trace(go.Scatter(x=np.concatenate([years_sp, years_sp[::-1]]),
                                              y=np.concatenate([Bpp_hi, Bpp_lo[::-1]]),
                                              fill="toself", fillcolor="rgba(99,110,250,0.2)",
                                              line=dict(color="rgba(0,0,0,0)"), name="90% CI"))
                fig_ret.add_trace(go.Scatter(x=years_sp, y=Bpp_md, mode="lines",
                                              line=dict(color=C_BLUE, width=2), name="Posterior median"))
                fig_ret.add_trace(go.Scatter(x=years_sp, y=Ba_sp, mode="markers",
                                              marker=dict(size=10, color="black", symbol="x"),
                                              name="Observed Ba"))
                fig_ret.update_layout(title="Posterior predictive retrodiction — PPL",
                                       xaxis_title="Year", yaxis_title="B (m w.e./yr)",
                                       height=340, margin=dict(t=50, b=20))
                st.plotly_chart(fig_ret, use_container_width=True)
            else:
                st.info("Click **Run PPL Metropolis** to fit the model.")

    # ── Section 4: PyMC inference ──────────────────────────────────────────────
    with st.expander("⚡ Section 4 — Fit with PyMC (NUTS)", expanded=False):
        p4L, p4R = st.columns([1, 2])
        with p4L:
            st.markdown("### The same model in PyMC")
            st.code("""\
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as pymc_model:
    # Priors (same as PPL model)
    mu_pm    = pm.HalfNormal('mu',    sigma=1.0)
    c_pm     = pm.HalfNormal('c',     sigma=1.0)
    T0_pm    = pm.Normal('T0',    mu=2, sigma=1.5)
    tau_pm   = pm.Exponential('tau',  lam=1/20)
    V0_pm    = pm.Normal('V0',    mu=10, sigma=5)
    sigma_pm = pm.HalfNormal('sigma', sigma=0.5)

    # Forward Euler ODE using pytensor ops
    # Python loop builds the computational graph once
    T_pt = pt.as_tensor_variable(T_melt)
    P_pt = pt.as_tensor_variable(P_acc)
    V_curr = V0_pm
    B_list = []
    for t in range(n_years):
        melt_t  = mu_pm * pt.maximum(T_pt[t] - T0_pm, 0.0)
        accum_t = c_pm  * P_pt[t]
        dV_t    = accum_t - melt_t - V_curr / tau_pm
        B_list.append(dV_t)
        V_curr  = V_curr + dV_t
    B_sim_pt = pt.stack(B_list)

    # Likelihood
    pm.Normal('B_obs', mu=B_sim_pt, sigma=sigma_pm,
              observed=B_obs)

    # NUTS (No U-Turn Sampler)
    trace = pm.sample(1000, tune=1000, target_accept=0.9,
                      progressbar=False, chains=2)
""", language="python")
            pymc_n    = st.slider("Draws per chain", 500, 2000, 1000, 250, key="gl_pymc_n")
            pymc_tune = st.slider("Tuning steps",    500, 2000, 1000, 250, key="gl_pymc_tune")
            run_pymc  = st.button("Run PyMC NUTS", type="primary", key="gl_pymc_run")

        with p4R:
            if run_pymc:
                try:
                    import pymc as pm
                    import pytensor.tensor as pt
                    import arviz as az
                    import logging
                    logging.getLogger("pymc").setLevel(logging.ERROR)
                    warnings.filterwarnings("ignore")

                    T_melt_np = T_melt_sp.astype("float64")
                    P_acc_np  = P_acc_sp.astype("float64")
                    B_obs_np  = Ba_sp.astype("float64")
                    ny        = n_years_sp

                    with pm.Model() as pm_model:
                        mu_pm    = pm.HalfNormal("mu",    sigma=1.0)
                        c_pm     = pm.HalfNormal("c",     sigma=1.0)
                        T0_pm    = pm.Normal("T0",    mu=2.0, sigma=1.5)
                        tau_pm   = pm.Exponential("tau",  lam=1.0/20.0)
                        V0_pm    = pm.Normal("V0",    mu=10.0, sigma=5.0)
                        sigma_pm = pm.HalfNormal("sigma", sigma=0.5)

                        T_pt = pt.as_tensor_variable(T_melt_np)
                        P_pt = pt.as_tensor_variable(P_acc_np)
                        V_curr = V0_pm
                        B_list_pt = []
                        for t in range(ny):
                            melt_t  = mu_pm * pt.maximum(T_pt[t] - T0_pm, 0.0)
                            accum_t = c_pm  * P_pt[t]
                            dV_t    = accum_t - melt_t - V_curr / tau_pm
                            B_list_pt.append(dV_t)
                            V_curr  = V_curr + dV_t
                        B_sim_pt = pt.stack(B_list_pt)

                        pm.Normal("B_obs", mu=B_sim_pt, sigma=sigma_pm, observed=B_obs_np)

                        with st.spinner(f"Running PyMC NUTS ({int(pymc_n)} draws, {int(pymc_tune)} tune)…"):
                            trace_pm = pm.sample(int(pymc_n), tune=int(pymc_tune),
                                                  target_accept=0.9, progressbar=False,
                                                  chains=2, cores=1,
                                                  return_inferencedata=True)

                    st.session_state["glacier_pymc_res"] = dict(trace=trace_pm)

                except Exception as e:
                    st.error(f"PyMC error: {e}")

            res_pm = st.session_state["glacier_pymc_res"]
            if res_pm:
                import arviz as az
                trace_pm = res_pm["trace"]
                param_names_pm = ["mu", "c", "T0", "tau", "V0", "sigma"]

                st.markdown("### ArviZ summary")
                summary_df = az.summary(trace_pm, var_names=param_names_pm)
                st.dataframe(summary_df.style.format("{:.3f}"), use_container_width=True)

                st.markdown("### Posterior distributions (NUTS)")
                fig_az, axes_az = plt.subplots(2, 3, figsize=(13, 6))
                fig_az.suptitle("PyMC NUTS — Glacier Posterior", fontsize=12, fontweight="bold")
                colors_az = [C_BLUE, C_ORANGE, C_GREEN, C_RED, "#9467bd", "#8c564b"]
                for idx, k in enumerate(param_names_pm):
                    r, c2 = divmod(idx, 3)
                    chain_az = trace_pm.posterior[k].values.flatten()
                    axes_az[r, c2].hist(chain_az, bins=50, color=colors_az[idx],
                                         alpha=0.7, density=True)
                    axes_az[r, c2].axvline(chain_az.mean(), color="black", lw=1.5,
                                            label=f"mean={chain_az.mean():.3f}")
                    axes_az[r, c2].set_title(k, fontsize=10)
                    axes_az[r, c2].legend(fontsize=7)
                plt.tight_layout()
                st.pyplot(fig_az)
                plt.close(fig_az)

                # Posterior predictive retrodiction
                mu_az  = trace_pm.posterior["mu"].values.flatten()
                c_az   = trace_pm.posterior["c"].values.flatten()
                T0_az  = trace_pm.posterior["T0"].values.flatten()
                tau_az = trace_pm.posterior["tau"].values.flatten()
                V0_az  = trace_pm.posterior["V0"].values.flatten()
                idx_samp = np.random.choice(len(mu_az), 200, replace=False)
                B_ppc = np.array([
                    glacier_ode(mu_az[i], c_az[i], T0_az[i], tau_az[i], V0_az[i],
                                T_melt_sp, P_acc_sp)
                    for i in idx_samp
                ])
                Bppc_lo = np.percentile(B_ppc, 5, axis=0)
                Bppc_hi = np.percentile(B_ppc, 95, axis=0)
                Bppc_md = np.percentile(B_ppc, 50, axis=0)

                fig_ret2 = go.Figure()
                fig_ret2.add_trace(go.Scatter(
                    x=np.concatenate([years_sp, years_sp[::-1]]),
                    y=np.concatenate([Bppc_hi, Bppc_lo[::-1]]),
                    fill="toself", fillcolor="rgba(239,85,59,0.2)",
                    line=dict(color="rgba(0,0,0,0)"), name="90% CI"))
                fig_ret2.add_trace(go.Scatter(x=years_sp, y=Bppc_md, mode="lines",
                                               line=dict(color=C_RED, width=2), name="Posterior median"))
                fig_ret2.add_trace(go.Scatter(x=years_sp, y=Ba_sp, mode="markers",
                                               marker=dict(size=10, color="black", symbol="x"),
                                               name="Observed Ba"))
                fig_ret2.update_layout(title="Posterior predictive retrodiction — PyMC NUTS",
                                        xaxis_title="Year", yaxis_title="B (m w.e./yr)",
                                        height=340, margin=dict(t=50, b=20))
                st.plotly_chart(fig_ret2, use_container_width=True)
            else:
                st.info("Click **Run PyMC NUTS** to fit the model.")

    # ── Section 5: Comparison ──────────────────────────────────────────────────
    with st.expander("⚖️ Section 5 — Compare both samplers", expanded=False):
        res_ppl_c  = st.session_state["glacier_ppl_res"]
        res_pymc_c = st.session_state["glacier_pymc_res"]

        if res_ppl_c and res_pymc_c:
            import arviz as az
            st.markdown("### Posterior violin plots — PPL Metropolis vs PyMC NUTS")
            sgl_c = res_ppl_c["samples"]
            trace_c = res_pymc_c["trace"]
            compare_params = ["mu", "c", "T0", "tau"]

            fig_viol, axes_v = plt.subplots(1, len(compare_params), figsize=(13, 5))
            fig_viol.suptitle("PPL Metropolis (blue) vs PyMC NUTS (red) — posterior marginals",
                               fontsize=11, fontweight="bold")
            for idx, k in enumerate(compare_params):
                ppl_ch  = np.array([s[k] for s in sgl_c])
                pymc_ch = trace_c.posterior[k].values.flatten()
                ax_v = axes_v[idx]
                parts = ax_v.violinplot([ppl_ch, pymc_ch], positions=[0, 1],
                                         showmedians=True, showextrema=False)
                parts["bodies"][0].set_facecolor(C_BLUE); parts["bodies"][0].set_alpha(0.6)
                parts["bodies"][1].set_facecolor(C_RED);  parts["bodies"][1].set_alpha(0.6)
                ax_v.set_xticks([0, 1]); ax_v.set_xticklabels(["PPL\nMetropolis", "PyMC\nNUTS"])
                ax_v.set_title(k, fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_viol)
            plt.close(fig_viol)

            st.markdown("### Side-by-side posterior means")
            cmp_data = {}
            for k in compare_params:
                ppl_v  = np.array([s[k] for s in sgl_c])
                pymc_v = trace_c.posterior[k].values.flatten()
                cmp_data[k] = {
                    "PPL mean": round(float(ppl_v.mean()), 3),
                    "PPL std":  round(float(ppl_v.std()),  3),
                    "NUTS mean": round(float(pymc_v.mean()), 3),
                    "NUTS std":  round(float(pymc_v.std()),  3),
                }
            st.dataframe(pd.DataFrame(cmp_data).T, use_container_width=True)

            st.markdown("""
### What did each sampler struggle with?

**PPL Metropolis**
- Proposes each parameter *independently* with a fixed Gaussian, so correlated
  parameters (e.g. μ and T₀ both affect melt) cause the acceptance rate to drop
  unless `proposal_std` is tuned carefully.
- Each evaluation calls the forward Euler ODE in Python — slow enough that
  5000 samples takes several seconds.  HMC would need numerical gradients with
  `2·n_params` ODE calls per leapfrog step, making it even slower here.
- The chain can get stuck near τ ≈ 0 or σ ≈ 0 without constrained proposals.

**PyMC NUTS**
- The pytensor computation graph allows exact gradient computation via autodiff —
  each NUTS step costs one gradient call (not 2·n_params × n_leapfrog evaluations).
- NUTS self-tunes the step size and number of leapfrog steps during warmup, so
  no manual tuning of ε or L is needed.
- The No-U-Turn stopping criterion prevents energy-wasting U-turns, giving
  nearly uncorrelated draws — low ACF lag-1, small ESS/draw ratio.
- Practical result: 2 × 1000 NUTS draws typically outperform 5000 Metropolis
  samples in effective sample size for this model.

**When would you choose the PPL over PyMC?**
For teaching, rapid prototyping, or highly custom models that don't fit any
standard distribution family.  PyMC wins when you need production-quality
inference on a well-structured model.
""")
        else:
            st.info("Run both **Section 3** (PPL) and **Section 4** (PyMC) first to see the comparison.")
