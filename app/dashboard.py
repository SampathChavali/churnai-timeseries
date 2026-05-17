"""
app/dashboard.py
----------------
ChurnAI Time-Series console — Radiant-inspired UI.

Theme: pure black background, white type, orange/red gradient accents,
sharp 2-4px corner radii, generous whitespace, massive headings, vertical
orange "light bar" accents.

Pages:
    HOME                    Hero, KPIs, "Ask the data" CTA
    ASK THE DATA            Chatbot grounded in the live CSVs
    CUSTOMER DEEP DIVE      Per-customer 12-month trajectory + attention
    PREDICT                 Live LSTM scoring with interactive sliders
    COHORTS                 High-risk vs low-risk cohort comparison
    MODEL                   Test metrics + training curves

Run:
    streamlit run app/dashboard.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as C
from models import lstm_encoder as M
from models.chat_engine import ChatEngine


# ===========================================================================
# Page config + theme
# ===========================================================================
st.set_page_config(
    page_title="ChurnAI · Time-Series Console",
    page_icon="◢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Radiant-inspired CSS -------------------------------------------------
CSS = """
<style>
:root {
    --bg:       #000;
    --bg-2:     #0a0a0a;
    --bg-3:     #131313;
    --line:     #1f1f1f;
    --text:     #f5f5f5;
    --text-dim: #8a8a8a;
    --text-2:   #b8b8b8;
    --orange:   #ff4d1c;
    --orange-2: #ff7a45;
    --orange-3: #ffae87;
    --green:    #4ade80;
    --yellow:   #f59e0b;
}

/* ---- App shell ---- */
.stApp {
    background: var(--bg);
    color: var(--text);
}
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1300px;
}
/* Hide Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ---- Typography ---- */
html, body, [class*="css"] {
    font-family: -apple-system, "Inter", "Helvetica Neue", Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
}
h1, h2, h3, h4 {
    color: var(--text);
    letter-spacing: -0.02em;
    font-weight: 600;
    line-height: 1.05;
}
h1 { font-size: 4.2rem; font-weight: 300; }
h2 { font-size: 2.2rem; font-weight: 400; }
h3 { font-size: 1.3rem; font-weight: 500; }
p, span, div { color: var(--text-2); }
.dim { color: var(--text-dim); font-size: 0.86rem; }
.kicker {
    color: var(--orange);
    font-size: 0.75rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    font-weight: 600;
}
.accent {
    background: linear-gradient(90deg, var(--orange) 0%, var(--orange-2) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 400;
}

/* ---- Top nav bar (rendered via a single radio) ---- */
div[data-testid="stRadio"][role="radiogroup"] > div {
    flex-direction: row !important;
    gap: 4px;
    border-bottom: 1px solid var(--line);
    padding-bottom: 14px;
    margin-bottom: 32px;
}
div[data-testid="stRadio"] label {
    background: transparent;
    border: 1px solid transparent;
    padding: 8px 18px;
    border-radius: 2px;
    color: var(--text-dim);
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-weight: 600;
    cursor: pointer;
    transition: color .15s ease, background .15s ease, border-color .15s ease;
}
div[data-testid="stRadio"] label:hover {
    color: var(--text);
}
div[data-testid="stRadio"] label:has(input:checked) {
    color: var(--text);
    border-color: var(--orange);
}
div[data-testid="stRadio"] label input { display: none; }

/* Hide the radio label text "Navigate" */
div[data-testid="stRadio"] > label[data-testid="stWidgetLabel"] { display: none !important; }

/* ---- Buttons ---- */
.stButton > button {
    background: var(--orange);
    color: #fff;
    border: none;
    border-radius: 2px;
    padding: 0.65rem 1.4rem;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    transition: background .15s ease, transform .1s ease;
}
.stButton > button:hover {
    background: var(--orange-2);
    transform: translateY(-1px);
}
.stButton > button:focus { box-shadow: none; outline: 1px solid var(--orange); }

/* Secondary button variant — use st.button(..., type="secondary") */
.stButton button[kind="secondary"] {
    background: transparent;
    color: var(--text);
    border: 1px solid var(--line);
}
.stButton button[kind="secondary"]:hover {
    border-color: var(--orange);
    color: var(--orange);
}

/* ---- Metric cards ---- */
div[data-testid="stMetric"] {
    background: var(--bg-2);
    border: 1px solid var(--line);
    border-left: 3px solid var(--orange);
    border-radius: 2px;
    padding: 22px 22px;
}
div[data-testid="stMetricLabel"] {
    color: var(--text-dim) !important;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-size: 0.68rem !important;
    font-weight: 600;
}
div[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-weight: 300;
    font-size: 2.4rem !important;
}
div[data-testid="stMetricDelta"] {
    color: var(--text-dim) !important;
    font-size: 0.78rem !important;
}

/* ---- Tables ---- */
[data-testid="stDataFrame"] {
    border-radius: 2px;
    border: 1px solid var(--line);
    overflow: hidden;
}

/* ---- Inputs ---- */
.stTextInput input, .stSelectbox > div > div, .stMultiSelect > div > div,
.stSlider {
    background: var(--bg-2) !important;
    color: var(--text) !important;
    border-radius: 2px !important;
}
.stTextInput input {
    border: 1px solid var(--line) !important;
    padding: 0.7rem 1rem !important;
}
.stTextInput input:focus { border-color: var(--orange) !important; box-shadow: none !important; }

/* ---- Custom card / pill ---- */
.card {
    background: var(--bg-2);
    border: 1px solid var(--line);
    border-radius: 2px;
    padding: 28px 28px;
}
.card-orange {
    border-left: 3px solid var(--orange);
}
.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 2px;
    font-size: 0.66rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 600;
}
.pill-orange { background: rgba(255,77,28,0.12); color: var(--orange); border: 1px solid rgba(255,77,28,0.4); }
.pill-yellow { background: rgba(245,158,11,0.12); color: var(--yellow); border: 1px solid rgba(245,158,11,0.4); }
.pill-green  { background: rgba(74,222,128,0.12); color: var(--green);  border: 1px solid rgba(74,222,128,0.4); }

/* ---- Hero ---- */
.hero-wrap {
    position: relative;
    padding: 38px 0 38px 0;
    border-bottom: 1px solid var(--line);
    margin-bottom: 40px;
}
.hero-title {
    font-size: 5.2rem;
    font-weight: 300;
    line-height: 1.0;
    color: var(--text);
    letter-spacing: -0.035em;
    margin: 0;
}
.hero-title em {
    font-style: normal;
    background: linear-gradient(90deg, var(--orange) 0%, var(--orange-2) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: var(--text-dim);
    font-size: 1.05rem;
    line-height: 1.6;
    max-width: 720px;
    margin-top: 26px;
}
.hero-bars {
    position: absolute; top: 0; right: 0; bottom: 0;
    width: 40%;
    display: flex; justify-content: flex-end; gap: 28px;
    pointer-events: none; opacity: 0.55;
}
.hero-bars span {
    display: block; width: 1px; height: 100%;
    background: linear-gradient(180deg, transparent 0%, var(--orange) 50%, transparent 100%);
    box-shadow: 0 0 18px 1px rgba(255,77,28,0.4);
}

/* ---- Section label (small kicker + thin orange bar) ---- */
.section-label {
    display: flex; align-items: center; gap: 14px;
    margin: 38px 0 14px 0;
    color: var(--orange);
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    font-weight: 600;
}
.section-label::before {
    content: "";
    display: inline-block;
    width: 36px; height: 1px;
    background: var(--orange);
}

/* ---- Chat ---- */
.chat-wrap {
    background: var(--bg-2);
    border: 1px solid var(--line);
    border-radius: 2px;
    padding: 8px 0;
    max-height: 620px;
    overflow-y: auto;
}
.chat-row { padding: 16px 28px; display: flex; gap: 16px; }
.chat-row.user   { background: transparent; }
.chat-row.assistant { background: rgba(255,77,28,0.03); border-left: 2px solid var(--orange); }
.chat-avatar {
    flex: 0 0 36px; width: 36px; height: 36px;
    border-radius: 2px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.78rem; letter-spacing: 0.06em;
}
.chat-avatar.user { background: var(--bg-3); color: var(--text-dim); border: 1px solid var(--line); }
.chat-avatar.ai   { background: var(--orange); color: #000; }
.chat-body { flex: 1; }
.chat-body .role {
    color: var(--text-dim); font-size: 0.68rem;
    letter-spacing: 0.18em; text-transform: uppercase; font-weight: 600;
    margin-bottom: 6px;
}
.chat-body .msg p, .chat-body .msg li {
    color: var(--text); font-size: 0.96rem; line-height: 1.55;
}
.chat-body code {
    background: var(--bg-3); color: var(--orange-2);
    padding: 2px 6px; border-radius: 2px; font-size: 0.85em;
}
.chat-intent {
    display: inline-block;
    color: var(--orange);
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 6px;
}

/* ---- Suggestion chips ---- */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 18px 0; }

/* Make Streamlit form-submit / chip buttons match */
button[kind="secondary"][data-testid="baseButton-secondary"] {
    background: transparent !important;
    color: var(--text-2) !important;
    border: 1px solid var(--line) !important;
    border-radius: 2px !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-size: 0.82rem !important;
    padding: 6px 12px !important;
}
button[kind="secondary"][data-testid="baseButton-secondary"]:hover {
    color: var(--orange) !important;
    border-color: var(--orange) !important;
}

/* ---- Risk pills ---- */
.risk-high   { background: rgba(255,77,28,0.12); color: var(--orange); border: 1px solid rgba(255,77,28,0.4); }
.risk-medium { background: rgba(245,158,11,0.12); color: var(--yellow); border: 1px solid rgba(245,158,11,0.4); }
.risk-low    { background: rgba(74,222,128,0.12); color: var(--green);  border: 1px solid rgba(74,222,128,0.4); }
.tag-risk {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 2px;
    font-size: 0.66rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 700;
}

/* ---- Brand bar on top ---- */
.brand-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 0 12px 0;
    margin-bottom: 0;
}
.brand-bar .logo {
    font-size: 1.0rem;
    font-weight: 700;
    letter-spacing: 0.28em;
    color: var(--text);
}
.brand-bar .logo span { color: var(--orange); }
.brand-bar .status {
    display: inline-flex; align-items: center; gap: 8px;
    color: var(--text-dim);
    font-size: 0.7rem;
    letter-spacing: 0.18em; text-transform: uppercase;
}
.brand-bar .status::before {
    content: ""; width: 6px; height: 6px; border-radius: 50%;
    background: var(--green); box-shadow: 0 0 8px var(--green);
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ===========================================================================
# Data loaders
# ===========================================================================
@st.cache_data(show_spinner=False)
def load_preds() -> Optional[pd.DataFrame]:
    return pd.read_csv(C.PREDICTIONS_CSV) if os.path.exists(C.PREDICTIONS_CSV) else None


@st.cache_data(show_spinner=False)
def load_ts() -> Optional[pd.DataFrame]:
    return pd.read_csv(C.TIMESERIES_CSV) if os.path.exists(C.TIMESERIES_CSV) else None


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    if not os.path.exists(C.METRICS_JSON):
        return {}
    with open(C.METRICS_JSON) as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    if not (os.path.exists(C.LSTM_MODEL_PATH) and os.path.exists(C.TS_SCALER_PATH)):
        return None, None
    return M.load_model(), M.load_scaler()


@st.cache_resource(show_spinner=False)
def load_engine(_preds_hash: str, _ts_hash: str, _met_hash: str) -> ChatEngine:
    return ChatEngine(load_preds(), load_ts(), load_metrics())


preds   = load_preds()
ts_full = load_ts()
metrics = load_metrics()
model, scaler = load_model_and_scaler()

# Pre-flight
if preds is None or ts_full is None or model is None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero-wrap"><h1 class="hero-title">Setup<em> required.</em></h1>'
                '<p class="hero-sub">No trained model or predictions found. '
                'Run the training pipeline first, then refresh this page.</p></div>',
                unsafe_allow_html=True)
    st.code("python train.py", language="bash")
    st.stop()

engine = load_engine(str(len(preds)), str(len(ts_full)), str(len(metrics)))


# ===========================================================================
# Plotly default styling
# ===========================================================================
def _layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif",
                  color="#b8b8b8", size=12),
        margin=dict(l=24, r=20, t=46, b=40),
        height=height,
        title_font=dict(color="#f5f5f5", size=14, family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(color="#b8b8b8")),
        xaxis=dict(gridcolor="#1f1f1f", zerolinecolor="#1f1f1f",
                   linecolor="#1f1f1f", tickcolor="#1f1f1f"),
        yaxis=dict(gridcolor="#1f1f1f", zerolinecolor="#1f1f1f",
                   linecolor="#1f1f1f", tickcolor="#1f1f1f"),
    )
    return fig


def chart_from_spec(spec: dict) -> go.Figure:
    kind = spec.get("kind")
    title = spec.get("title", "")

    if kind == "donut":
        fig = go.Figure(go.Pie(
            labels=spec["labels"], values=spec["values"],
            marker=dict(colors=spec.get("colors")),
            hole=0.65, textinfo="label+percent",
            textfont=dict(color="#f5f5f5"),
        ))
        fig.update_layout(showlegend=False, title=title)
        return _layout(fig, height=360)

    if kind == "bar":
        color = spec.get("color")
        if isinstance(color, list):
            marker = dict(color=color)
        else:
            marker = dict(color=color or "#ff4d1c")
        fig = go.Figure(go.Bar(
            x=spec["x"], y=spec["y"], marker=marker,
        ))
        fig.update_layout(title=title,
                          xaxis_title=spec.get("xaxis", ""),
                          yaxis_title=spec.get("yaxis", ""))
        return _layout(fig)

    if kind == "hbar":
        fig = go.Figure(go.Bar(
            x=spec["x"], y=spec["y"], orientation="h",
            marker=dict(color=spec.get("color", "#ff4d1c")),
        ))
        fig.update_layout(title=title,
                          xaxis_title=spec.get("xaxis", ""),
                          yaxis_title=spec.get("yaxis", ""))
        return _layout(fig, height=max(360, 24 * len(spec["y"]) + 80))

    if kind == "lines":
        fig = go.Figure()
        palette = spec.get("colors") or ["#ff4d1c", "#ff7a45", "#ffae87", "#4ade80"]
        for i, (name, ys) in enumerate(spec["series"].items()):
            color = palette[i % len(palette)]
            fig.add_trace(go.Scatter(
                x=spec["x"], y=ys, mode="lines+markers",
                name=name, line=dict(color=color, width=3),
                marker=dict(size=6),
            ))
        fig.update_layout(title=title,
                          xaxis=dict(title=spec.get("xaxis", ""), dtick=1),
                          yaxis_title=spec.get("yaxis", ""))
        return _layout(fig)

    return _layout(go.Figure())


def trend_chart(ts_one: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_one["month"], y=ts_one["data_gb"], mode="lines+markers",
        name="Data (GB)", line=dict(color="#ff4d1c", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=ts_one["month"], y=ts_one["call_minutes"] / 10, mode="lines+markers",
        name="Calls (×10 min)", line=dict(color="#ff7a45", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=ts_one["month"], y=ts_one["login_count"], mode="lines+markers",
        name="Logins", line=dict(color="#ffae87", width=3),
    ))
    fig.add_trace(go.Bar(
        x=ts_one["month"], y=ts_one["support_tickets"],
        name="Tickets", yaxis="y2",
        marker=dict(color="rgba(255,77,28,0.45)"),
    ))
    first6 = ts_one["data_gb"].iloc[:6].mean()
    last3  = ts_one["data_gb"].iloc[-3:].mean()
    if first6 > 0 and last3 < 0.75 * first6:
        drop = (1 - last3 / first6) * 100
        fig.add_vrect(
            x0=9.5, x1=12.5, fillcolor="rgba(255,77,28,0.12)", line_width=0,
            annotation_text=f"usage ↓ {drop:.0f}% in last 3 mo",
            annotation_position="top left",
            annotation_font_color="#ff7a45",
        )
    fig.update_layout(
        title="12-month behavioral trajectory",
        xaxis=dict(title="Month", dtick=1),
        yaxis_title="Usage / logins",
        yaxis2=dict(title="Tickets", overlaying="y", side="right",
                    showgrid=False, color="#b8b8b8"),
        legend=dict(orientation="h", y=-0.18),
    )
    return _layout(fig, height=440)


def attention_chart(attn: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=list(range(1, len(attn) + 1)), y=attn,
        marker=dict(
            color=attn,
            colorscale=[[0.0, "#1f1f1f"], [1.0, "#ff4d1c"]],
        ),
    ))
    fig.update_layout(
        title="LSTM attention over months",
        xaxis=dict(title="Month", dtick=1),
        yaxis_title="weight",
    )
    return _layout(fig, height=240)


def prob_gauge(p: float) -> go.Figure:
    color = "#ff4d1c" if p >= C.RISK_HIGH else "#f59e0b" if p >= C.RISK_MEDIUM else "#4ade80"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number={"suffix": "%", "font": {"size": 38, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8a8a8a"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, C.RISK_MEDIUM * 100],          "color": "rgba(74,222,128,0.15)"},
                {"range": [C.RISK_MEDIUM * 100, C.RISK_HIGH * 100], "color": "rgba(245,158,11,0.15)"},
                {"range": [C.RISK_HIGH * 100, 100],          "color": "rgba(255,77,28,0.18)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f5f5f5"),
        height=240, margin=dict(l=10, r=10, t=20, b=20),
    )
    return fig


# ===========================================================================
# Top brand bar + nav
# ===========================================================================
st.markdown(
    '<div class="brand-bar">'
    '<div class="logo">CHURN<span>AI</span></div>'
    '<div class="status">Live</div>'
    '</div>',
    unsafe_allow_html=True,
)

PAGES = ["HOME", "ASK THE DATA", "DEEP DIVE", "PREDICT", "COHORTS", "MODEL"]

# Restore page choice from query state if a chat suggestion redirected here
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]

page = st.radio(
    "Navigate",
    PAGES,
    index=PAGES.index(st.session_state.page),
    key="nav_radio",
    horizontal=True,
    label_visibility="collapsed",
)
st.session_state.page = page


# ===========================================================================
# HOME
# ===========================================================================
def page_home() -> None:
    n_total = len(preds)
    n_high  = int((preds["risk_tier"] == "High").sum())
    n_med   = int((preds["risk_tier"] == "Medium").sum())
    rev_h   = float(preds.loc[preds["risk_tier"] == "High", "month12_charge"].fillna(0).sum())
    auc     = metrics.get("metrics", {}).get("test_auc", 0.0)

    st.markdown(
        f"""
        <div class="hero-wrap">
          <div class="hero-bars">
            <span style="height:78%"></span>
            <span style="height:64%"></span>
            <span style="height:88%"></span>
            <span style="height:52%"></span>
            <span style="height:72%"></span>
            <span style="height:46%"></span>
          </div>
          <div class="kicker">CHURN INTELLIGENCE · BiLSTM + ATTENTION</div>
          <h1 class="hero-title">We Are<br><em>Powered Foresight.</em></h1>
          <p class="hero-sub">
            ChurnAI reads 12 months of behavioral history for every customer
            and surfaces who is about to leave — before they raise a ticket.
            Trained on {n_total:,} customers, deployed locally, queryable in
            plain English.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers scored", f"{n_total:,}")
    c2.metric("High-risk flagged", f"{n_high:,}", f"{n_high / n_total:.1%} of base")
    c3.metric("Medium-risk", f"{n_med:,}")
    c4.metric("Monthly revenue at risk", f"${rev_h:,.0f}")

    st.markdown('<div class="section-label">Ask the model anything</div>', unsafe_allow_html=True)

    box_l, box_r = st.columns([2, 1])
    with box_l:
        st.markdown(
            """
            <div class="card card-orange">
              <div class="pill pill-orange">CHATBOT</div>
              <h2 style="margin:14px 0 8px 0;">Ask the data.</h2>
              <p style="color:var(--text-2); font-size:1.0rem; line-height:1.6;">
                Skip the dashboards. Type natural questions and get
                answers grounded in the live predictions and time-series
                CSVs — no hallucinations, every number is computed live.
              </p>
              <p class="dim" style="margin-top:14px;">
                Try: "Who are my top 10 at-risk customers?" ·
                "Compare churn by contract type" ·
                "What if I offer a 25% discount?"
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Streamlit will not let us mutate the nav radio's session-state key
        # in the script body once that widget has already been rendered.
        # The legal way to programmatically change a widget's value is to do
        # it inside an `on_click` callback, which runs *before* the next
        # rerun. After that, `st.button` already triggers a rerun on its own,
        # so we don't (and shouldn't) call `st.rerun()` manually.
        def _open_chatbot():
            st.session_state.nav_radio = "ASK THE DATA"
            st.session_state.page = "ASK THE DATA"

        st.button("OPEN THE CHATBOT",
                  key="open_chat_btn",
                  on_click=_open_chatbot)
    with box_r:
        st.markdown(
            f"""
            <div class="card">
              <div class="kicker">MODEL HEALTH</div>
              <h3 style="margin: 12px 0;">{auc:.3f} AUC</h3>
              <p class="dim">Held-out test set ({metrics.get('metrics', {}).get('n_test', 0)} customers).</p>
              <hr style="border:none; border-top:1px solid var(--line); margin:18px 0;">
              <div style="color:var(--text-2); font-size:0.86rem; line-height:1.8;">
                <div>Accuracy   <span style="float:right; color:var(--text);">
                    {metrics.get('metrics', {}).get('test_accuracy', 0):.3f}</span></div>
                <div>Recall     <span style="float:right; color:var(--text);">
                    {metrics.get('metrics', {}).get('test_recall', 0):.3f}</span></div>
                <div>Precision  <span style="float:right; color:var(--text);">
                    {metrics.get('metrics', {}).get('test_precision', 0):.3f}</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- About the data + how live data enters the system ---------------
    st.markdown('<div class="section-label">About this dataset</div>',
                unsafe_allow_html=True)
    info_l, info_r = st.columns(2)
    info_l.markdown(
        '<div class="card">'
        '<div class="kicker">CUSTOMER IDs</div>'
        '<h3 style="margin:10px 0 8px 0;">Format: <code>NNNN-AAAAA</code></h3>'
        '<p style="color:var(--text-2);font-size:0.92rem;line-height:1.6;">'
        'Customer IDs follow the Kaggle Telco convention: a 4-digit '
        'sequence number followed by 5 random uppercase letters '
        '(e.g.&nbsp;<code>0982-QHUUQ</code>). The dataset is fully '
        '<strong>synthetic but deterministic</strong> — each ID is hashed '
        'to seed a per-customer random generator, so re-running '
        '<code>python generate_data.py</code> always produces the same '
        'rows on any machine.'
        '</p>'
        '<p class="dim" style="margin-top:10px;">'
        'There is no PII here. The IDs are stand-ins for what a real '
        'telco would use internally.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    info_r.markdown(
        '<div class="card">'
        '<div class="kicker">WHERE DATA ENTERS THE SYSTEM</div>'
        '<h3 style="margin:10px 0 8px 0;">Three entry points</h3>'
        '<ol style="color:var(--text-2);font-size:0.92rem;line-height:1.7;padding-left:18px;">'
        '<li><strong>Batch scoring (default):</strong> <code>python train.py</code> '
        'reads <code>data/time_series.csv</code> and produces '
        '<code>data/predictions.csv</code>. Every page in this UI reads that file.</li>'
        '<li><strong>Single customer:</strong> the <em>PREDICT</em> page '
        'feeds 12 monthly snapshots straight into the loaded LSTM and '
        'scores instantly — no CSV needed.</li>'
        '<li><strong>New historical data:</strong> append rows to '
        '<code>data/time_series.csv</code> and re-run <code>train.py</code>.</li>'
        '</ol>'
        '<p class="dim" style="margin-top:6px;">'
        'The "Model loaded" indicator at the top means the trained '
        'BiLSTM is in memory and ready to score — it does <em>not</em> '
        'mean data is streaming.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Distribution</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1])
    donut = chart_from_spec({
        "kind": "donut",
        "labels": ["High", "Medium", "Low"],
        "values": [n_high, n_med, len(preds) - n_high - n_med],
        "colors": ["#ff4d1c", "#f59e0b", "#4ade80"],
        "title": "Risk distribution",
    })
    left.plotly_chart(donut, width="stretch")

    hist = go.Figure(go.Histogram(
        x=preds["churn_prob"], nbinsx=40,
        marker=dict(
            color=preds["churn_prob"],
            colorscale=[[0.0, "#4ade80"], [0.5, "#f59e0b"], [1.0, "#ff4d1c"]],
        ),
    ))
    hist.add_vline(x=C.RISK_HIGH,   line_dash="dot", line_color="#ff4d1c")
    hist.add_vline(x=C.RISK_MEDIUM, line_dash="dot", line_color="#f59e0b")
    hist.update_layout(title="Predicted churn probability",
                       xaxis_title="P(churn)", yaxis_title="customers",
                       bargap=0.04)
    right.plotly_chart(_layout(hist, height=360), width="stretch")


# ===========================================================================
# ASK THE DATA — the chatbot
# ===========================================================================
SUGGESTIONS = [
    "How many high-risk customers are there?",
    "Top 10 at-risk customers",
    "Compare churn by contract type",
    "Show me the 12-month data usage trend for churners",
    "How much monthly revenue is at risk?",
    "Which customers have the most support tickets?",
    "What is the model AUC and recall?",
    "What if I offer a 25% discount to high-risk customers?",
    "Tell me about customer 0001-CURLL",
    "Biggest drops in data usage",
]


def _render_message(role: str, content: str, intent: Optional[str] = None) -> None:
    """Render a single chat row.

    IMPORTANT: the HTML is emitted on a single line with no leading whitespace.
    Multi-line indented HTML triggers Streamlit's markdown parser to treat
    everything inside as a fenced code block, which causes the raw HTML to
    show as literal text on the page.
    """
    avatar_class = "ai" if role == "assistant" else "user"
    avatar_text  = "AI" if role == "assistant" else "YOU"
    intent_pill  = (f'<div class="chat-intent">{intent.replace("_", " ")}</div>'
                    if role == "assistant" and intent else "")

    import html as _html
    import re as _re

    safe = _html.escape(content)
    safe = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = _re.sub(r"`([^`]+?)`", r"<code>\1</code>", safe)
    safe = safe.replace("\n\n", "<br><br>").replace("\n", "<br>")

    block = (
        f'<div class="chat-row {role}">'
        f'<div class="chat-avatar {avatar_class}">{avatar_text}</div>'
        f'<div class="chat-body">'
        f'<div class="role">{role}</div>'
        f'{intent_pill}'
        f'<div class="msg"><p>{safe}</p></div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(block, unsafe_allow_html=True)


def page_ask() -> None:
    if "chat" not in st.session_state:
        st.session_state.chat = [{
            "role": "assistant",
            "intent": "greeting",
            "content": engine.ask("hi").text,
            "chart": None,
        }]

    st.markdown('<div class="kicker">CHATBOT</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="margin: 6px 0 4px 0;">Ask the data.</h2>'
                '<p class="dim" style="margin-bottom: 22px;">'
                'Every answer is computed live from the predictions and '
                'time-series CSVs. No external LLM, no hallucinations.</p>',
                unsafe_allow_html=True)

    left, right = st.columns([3, 1])

    # ---- RIGHT: live stats + actions ----
    with right:
        st.markdown(
            f"""
            <div class="card">
              <div class="kicker">LIVE STATS</div>
              <div style="margin-top:14px;color:var(--text-2);font-size:0.88rem;line-height:1.9;">
                <div>Customers     <span style="float:right;color:var(--text);">{len(preds):,}</span></div>
                <div>High risk     <span style="float:right;color:var(--orange);">
                    {(preds['risk_tier'] == 'High').sum():,}</span></div>
                <div>Medium risk   <span style="float:right;color:var(--yellow);">
                    {(preds['risk_tier'] == 'Medium').sum():,}</span></div>
                <div>Low risk      <span style="float:right;color:var(--green);">
                    {(preds['risk_tier'] == 'Low').sum():,}</span></div>
                <div>Avg P(churn)  <span style="float:right;color:var(--text);">
                    {preds['churn_prob'].mean()*100:.1f}%</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("CLEAR CHAT", type="secondary", key="clear_chat_btn"):
            st.session_state.chat = []
            st.rerun()

    # ---- LEFT: conversation + composer ----
    with left:
        # render history
        with st.container():
            st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
            for i, m in enumerate(st.session_state.chat):
                _render_message(m["role"], m["content"], m.get("intent"))
                if m.get("chart") is not None:
                    st.plotly_chart(chart_from_spec(m["chart"]),
                                    width="stretch", key=f"chart_{i}")
            st.markdown('</div>', unsafe_allow_html=True)

        # suggestions
        st.markdown('<div class="kicker" style="margin-top:18px;">SUGGESTED QUESTIONS</div>',
                    unsafe_allow_html=True)
        sugg_cols = st.columns(5)
        clicked: Optional[str] = None
        for i, s in enumerate(SUGGESTIONS):
            with sugg_cols[i % 5]:
                if st.button(s, key=f"sugg_{i}", type="secondary"):
                    clicked = s

        # composer
        with st.form("composer", clear_on_submit=True):
            cols = st.columns([5, 1])
            with cols[0]:
                q = st.text_input(
                    "ask",
                    placeholder="Ask anything about the dataset...",
                    label_visibility="collapsed",
                )
            with cols[1]:
                submitted = st.form_submit_button("ASK", width="stretch")

        query = clicked if clicked else (q if submitted and q else None)
        if query:
            st.session_state.chat.append({
                "role": "user", "content": query, "intent": None, "chart": None,
            })
            resp = engine.ask(query)
            st.session_state.chat.append({
                "role": "assistant",
                "content": resp.text,
                "intent": resp.intent,
                "chart": resp.chart,
            })
            st.rerun()


# ===========================================================================
# CUSTOMER DEEP DIVE
# ===========================================================================
def _risk_tag_html(tier: str) -> str:
    cls = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}.get(tier, "risk-low")
    return f'<span class="tag-risk {cls}">{tier.upper()} RISK</span>'


def page_deep_dive() -> None:
    st.markdown('<div class="kicker">PER-CUSTOMER ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="margin: 6px 0 22px 0;">Customer deep dive.</h2>', unsafe_allow_html=True)

    ids = preds.sort_values("churn_prob", ascending=False)["customerID"].tolist()
    cid = st.selectbox("Customer ID (highest risk first)", ids, index=0,
                       label_visibility="visible",
                       help=("IDs follow the Kaggle Telco convention "
                             "(4-digit prefix + 5 random letters). "
                             "All 2,000 customers in this demo are synthetic "
                             "but deterministic — the same ID always maps to "
                             "the same behaviour."))

    row = preds[preds["customerID"] == cid].iloc[0]
    ts_one = ts_full[ts_full["customerID"] == cid].sort_values("month")

    head_l, head_r = st.columns([2, 1])
    with head_l:
        st.markdown(
            f"""
            <div class="card card-orange">
              <div style="display:flex;align-items:center;gap:14px;">
                <div style="font-size:1.4rem;font-weight:600;color:var(--text);">{cid}</div>
                {_risk_tag_html(row['risk_tier'])}
              </div>
              <p class="dim" style="margin-top:8px;">
                Contract: <span style="color:var(--text);">{row['contract']}</span> ·
                Internet: <span style="color:var(--text);">{row['internetService']}</span> ·
                Tenure: <span style="color:var(--text);">{int(row['tenure_months'])} months</span> ·
                Truth: <span style="color:var(--text);">{'CHURNED' if row['true_label'] == 1 else 'RETAINED'}</span>
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with head_r:
        st.plotly_chart(prob_gauge(float(row["churn_prob"])), width="stretch")

    # live attention
    _, X_raw, _ = M.build_sequences(ts_one.assign(churn_label=int(row["true_label"])))
    X = M.apply_scaler(X_raw, scaler)
    _, _, attns = M.predict_with_attention(model, X)
    attn = attns[0]

    st.plotly_chart(trend_chart(ts_one), width="stretch")
    st.plotly_chart(attention_chart(attn), width="stretch")

    st.markdown('<div class="section-label">First 6 months vs last 3 months</div>',
                unsafe_allow_html=True)
    first6 = ts_one.iloc[:6][C.TS_FEATURES].mean()
    last3  = ts_one.iloc[-3:][C.TS_FEATURES].mean()
    deltas = pd.DataFrame({
        "feature": C.TS_FEATURES,
        "first 6 mo avg": [round(float(first6[c]), 2) for c in C.TS_FEATURES],
        "last 3 mo avg":  [round(float(last3[c]), 2) for c in C.TS_FEATURES],
        "Δ %": [round(float((last3[c] - first6[c]) / max(first6[c], 1e-6) * 100), 1)
                for c in C.TS_FEATURES],
    })
    st.dataframe(deltas, hide_index=True, width="stretch")


# ===========================================================================
# PREDICT
# ===========================================================================
def page_predict() -> None:
    st.markdown('<div class="kicker">LIVE INFERENCE</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="margin: 6px 0 22px 0;">Score a new customer.</h2>',
                unsafe_allow_html=True)

    preset = st.radio(
        "Quick preset",
        ["Custom", "Churner profile", "Retained profile"],
        horizontal=True, key="predict_preset",
    )

    if preset == "Churner profile":
        data_gb        = np.array([24, 23, 24, 22, 23, 22, 19, 17, 14, 11, 8, 6], dtype=float)
        call_minutes   = np.array([180, 175, 178, 170, 172, 168, 150, 140, 120, 100, 80, 65], dtype=float)
        monthly_charge = np.array([72, 72, 73, 74, 74, 75, 76, 77, 78, 79, 80, 81], dtype=float)
        support_tickets = np.array([0, 0, 1, 0, 1, 0, 1, 2, 2, 3, 3, 4], dtype=float)
        login_count    = np.array([26, 25, 26, 24, 25, 23, 19, 16, 12, 8, 5, 3], dtype=float)
        service_outages = np.array([0, 0, 0, 0, 1, 0, 1, 1, 2, 2, 3, 3], dtype=float)
    elif preset == "Retained profile":
        data_gb        = np.linspace(18, 22, 12)
        call_minutes   = np.linspace(150, 170, 12)
        monthly_charge = np.full(12, 58.0)
        support_tickets = np.zeros(12)
        login_count    = np.linspace(24, 28, 12)
        service_outages = np.zeros(12)
    else:
        data_gb        = np.full(12, 20.0)
        call_minutes   = np.full(12, 150.0)
        monthly_charge = np.full(12, 65.0)
        support_tickets = np.zeros(12)
        login_count    = np.full(12, 22.0)
        service_outages = np.zeros(12)

    st.markdown('<div class="section-label">Months 10 — 11 — 12 (tune these)</div>',
                unsafe_allow_html=True)
    cols = st.columns(3)
    for i, m in enumerate([10, 11, 12]):
        with cols[i]:
            st.markdown(f"<div class='kicker'>MONTH {m}</div>", unsafe_allow_html=True)
            data_gb[m - 1]         = st.slider(f"Data (GB)",     0.0, 50.0,  float(data_gb[m - 1]),         0.5, key=f"d{m}")
            call_minutes[m - 1]    = st.slider(f"Calls (min)",   0.0, 400.0, float(call_minutes[m - 1]),    5.0, key=f"c{m}")
            monthly_charge[m - 1]  = st.slider(f"Charge ($)",    0.0, 200.0, float(monthly_charge[m - 1]),  1.0, key=f"ch{m}")
            support_tickets[m - 1] = st.slider(f"Tickets",       0,   10,    int(support_tickets[m - 1]),  1,    key=f"t{m}")
            login_count[m - 1]     = st.slider(f"Logins",        0,   60,    int(login_count[m - 1]),       1,    key=f"l{m}")
            service_outages[m - 1] = st.slider(f"Outages",       0,   5,     int(service_outages[m - 1]),   1,    key=f"o{m}")

    seq = np.stack([
        data_gb, call_minutes, monthly_charge,
        support_tickets, login_count, service_outages,
    ], axis=1)[np.newaxis, :, :].astype(np.float32)
    X = M.apply_scaler(seq, scaler)
    probs, _, attns = M.predict_with_attention(model, X)
    p = float(probs[0]); attn = attns[0]
    tier = "High" if p >= C.RISK_HIGH else "Medium" if p >= C.RISK_MEDIUM else "Low"

    df_new = pd.DataFrame({
        "month": np.arange(1, C.SEQ_LEN + 1),
        "data_gb": data_gb, "call_minutes": call_minutes,
        "monthly_charge": monthly_charge,
        "support_tickets": support_tickets,
        "login_count": login_count,
        "service_outages": service_outages,
    })

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(trend_chart(df_new), width="stretch")
        st.plotly_chart(attention_chart(attn), width="stretch")
    with right:
        st.markdown(
            f"""
            <div class="card card-orange" style="text-align:center;">
              <div class="kicker">PREDICTED RISK</div>
              <div style="font-size:3rem;font-weight:300;color:var(--text);margin:8px 0;">
                {p*100:.1f}%
              </div>
              {_risk_tag_html(tier)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(prob_gauge(p), width="stretch")
        st.markdown(
            f"""
            <div class="card">
              <div class="kicker">ATTENTION PEAK</div>
              <div style="margin-top:8px;color:var(--text);font-size:1.1rem;">
                Month {int(np.argmax(attn)) + 1}
                <span class="dim" style="font-size:0.85rem;">({attn.max():.2f})</span>
              </div>
              <p class="dim" style="margin-top:8px;">
                The LSTM gave this month the heaviest weight in its decision.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# COHORTS
# ===========================================================================
def page_cohorts() -> None:
    st.markdown('<div class="kicker">COHORT INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="margin: 6px 0 22px 0;">High risk vs low risk.</h2>',
                unsafe_allow_html=True)

    high_ids = preds.loc[preds["risk_tier"] == "High", "customerID"]
    low_ids  = preds.loc[preds["risk_tier"] == "Low",  "customerID"]

    feature = st.selectbox("Feature", C.TS_FEATURES, index=0)
    avg_high = ts_full[ts_full["customerID"].isin(high_ids)].groupby("month")[feature].mean()
    avg_low  = ts_full[ts_full["customerID"].isin(low_ids)].groupby("month")[feature].mean()

    fig = chart_from_spec({
        "kind": "lines",
        "x": list(range(1, C.SEQ_LEN + 1)),
        "series": {
            f"High risk (n={len(high_ids)})": avg_high.tolist(),
            f"Low risk (n={len(low_ids)})":  avg_low.tolist(),
        },
        "colors": ["#ff4d1c", "#4ade80"],
        "title": f"Average {feature} by month",
        "xaxis": "month", "yaxis": feature,
    })
    st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="section-label">Risk mix by contract</div>', unsafe_allow_html=True)
    cross = pd.crosstab(preds["contract"], preds["risk_tier"]).reindex(
        columns=["High", "Medium", "Low"], fill_value=0
    )
    bar = go.Figure()
    for tier, color in zip(["High", "Medium", "Low"], ["#ff4d1c", "#f59e0b", "#4ade80"]):
        if tier in cross.columns:
            bar.add_trace(go.Bar(x=cross.index, y=cross[tier],
                                 name=tier, marker=dict(color=color)))
    bar.update_layout(barmode="stack", title="Customers by contract & tier",
                      xaxis_title="Contract", yaxis_title="customers")
    st.plotly_chart(_layout(bar), width="stretch")

    st.markdown('<div class="section-label">Snapshot by predicted tier</div>',
                unsafe_allow_html=True)
    snap = (preds.groupby("risk_tier")[
                ["churn_prob", "month12_data_gb", "month12_charge",
                 "month12_tickets", "month12_logins"]]
                  .mean(numeric_only=True)
                  .round(2)
                  .reindex(["High", "Medium", "Low"]))
    st.dataframe(snap, width="stretch")


# ===========================================================================
# MODEL
# ===========================================================================
def page_model() -> None:
    st.markdown('<div class="kicker">MODEL CARD</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="margin: 6px 0 22px 0;">BiLSTM with attention.</h2>',
                unsafe_allow_html=True)

    m = metrics.get("metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test AUC-ROC",   f"{m.get('test_auc', 0):.3f}")
    c2.metric("Test Accuracy",  f"{m.get('test_accuracy', 0):.3f}")
    c3.metric("Test Recall",    f"{m.get('test_recall', 0):.3f}")
    c4.metric("Test Precision", f"{m.get('test_precision', 0):.3f}")

    hist = m.get("history", {})
    if hist:
        epochs = list(range(1, len(hist.get("train_loss", [])) + 1))
        left, right = st.columns(2)
        loss = go.Figure()
        loss.add_trace(go.Scatter(x=epochs, y=hist["train_loss"], mode="lines+markers",
                                  name="train", line=dict(color="#ff4d1c", width=3)))
        loss.add_trace(go.Scatter(x=epochs, y=hist["val_loss"], mode="lines+markers",
                                  name="val", line=dict(color="#ff7a45", width=3)))
        loss.update_layout(title="Loss per epoch",
                           xaxis_title="epoch", yaxis_title="BCE w/ logits")
        left.plotly_chart(_layout(loss), width="stretch")

        auc = go.Figure(go.Scatter(
            x=epochs, y=hist["val_auc"], mode="lines+markers",
            line=dict(color="#ff4d1c", width=3),
        ))
        auc.update_layout(title="Validation AUC per epoch",
                          xaxis_title="epoch", yaxis_title="AUC-ROC")
        right.plotly_chart(_layout(auc), width="stretch")

    st.markdown('<div class="section-label">Architecture</div>', unsafe_allow_html=True)
    arch = pd.DataFrame([
        {"Layer": "Input",       "Shape": f"({C.SEQ_LEN}, {C.TS_INPUT_DIM})",
         "Notes": "12 monthly snapshots × 6 features"},
        {"Layer": "BiLSTM",      "Shape": f"hidden={C.LSTM_HIDDEN} × {C.LSTM_LAYERS} layers",
         "Notes": "bidirectional, dropout=0.3"},
        {"Layer": "Attention",   "Shape": "additive over T",
         "Notes": "learned query → softmax over time-steps"},
        {"Layer": "Projection",  "Shape": f"→ {C.LSTM_EMBED_DIM}-dim embedding",
         "Notes": "Linear + LayerNorm + Dropout"},
        {"Layer": "Classifier",  "Shape": "→ 1",
         "Notes": "Sigmoid · BCE w/ pos_weight=2.5"},
    ])
    st.dataframe(arch, hide_index=True, width="stretch")


# ===========================================================================
# Router
# ===========================================================================
if page == "HOME":
    page_home()
elif page == "ASK THE DATA":
    page_ask()
elif page == "DEEP DIVE":
    page_deep_dive()
elif page == "PREDICT":
    page_predict()
elif page == "COHORTS":
    page_cohorts()
elif page == "MODEL":
    page_model()
