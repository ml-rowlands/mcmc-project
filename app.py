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

for _k in ("mh_step", "hmc_step", "mh_res", "hmc_res"):
    if _k not in st.session_state:
        st.session_state[_k] = None if _k.endswith("res") else 0

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_ov, tab_mh, tab_hmc, tab_cmp = st.tabs([
    "Overview", "Metropolis-Hastings", "HMC & Leapfrog", "Comparison"
])

# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_ov:
    st.title("Markov Chain Monte Carlo — Interactive Explorer")
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown("""
### Why MCMC?

In Bayesian inference we often need samples from a posterior that we **cannot sample from directly**,
but can evaluate pointwise up to a normalizing constant:
""")
        st.latex(r"\pi(\theta \mid y) \;\propto\; p(y \mid \theta)\;\cdot\;p(\theta)")
        st.markdown("""
**MCMC constructs a Markov chain whose stationary distribution is π.**
After a *burn-in* period, the chain's positions are approximately distributed as π.

---
### Two algorithms

| | Metropolis-Hastings | Hamiltonian Monte Carlo |
|---|---|---|
| **Proposal** | Random Gaussian jump | Gradient-guided leap via Hamiltonian dynamics |
| **Mixing** | Slow (random walk) | Fast (directed exploration) |
| **Needs** | log π(q) | log π(q) **and** ∇log π(q) |
| **Key parameter** | proposal σ | step size ε, leapfrog steps L |

Use the tabs above to step through each algorithm interactively.
        """)
    with col_r:
        st.info("""
**How to use this app**

Each algorithm tab walks you through the idea **one concept at a time**.
Use **Next ▶** to advance, **◀** to review.

The **Comparison** tab lets you run both side-by-side.
        """)
        st.markdown("---")
        st.markdown("""
**Reading the diagnostics**

| | Good sign |
|---|---|
| Trace | Noisy, no trend |
| Histogram | Matches green density |
| ACF | Decays quickly to 0 |
| Acceptance | MH: 20–50 % · HMC: 60–90 % |
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
