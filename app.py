"""
app.py  –  Solar Radiation Forecasting Dashboard
=================================================
Streamlit deployment of the MSc dissertation pipeline.

Run with:
    streamlit run app.py

Or with custom port:
    streamlit run app.py --server.port 8501
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Solar Forecasting · West Africa",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root variables ── */
:root {
    --sun:      #F5A623;
    --amber:    #E8861A;
    --deep:     #0D1117;
    --panel:    #161B22;
    --border:   #30363D;
    --text:     #E6EDF3;
    --muted:    #8B949E;
    --good:     #3FB950;
    --warn:     #D29922;
    --bad:      #F85149;
    --accent:   #58A6FF;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--deep) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 2rem !important; max-width: 1400px; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a1200 0%, #2d1f00 40%, #0D1117 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '☀';
    position: absolute;
    right: 2rem; top: -0.5rem;
    font-size: 8rem;
    opacity: 0.06;
    line-height: 1;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--sun) !important;
    margin: 0 0 0.3rem !important;
    letter-spacing: -0.5px;
}
.hero p {
    color: var(--muted);
    font-size: 0.9rem;
    margin: 0 !important;
}
.hero .tag {
    display: inline-block;
    background: rgba(245,166,35,0.12);
    border: 1px solid rgba(245,166,35,0.3);
    color: var(--sun);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    margin-right: 6px;
    margin-top: 8px;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 140px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
    position: relative;
}
.metric-card .label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'Space Mono', monospace;
}
.metric-card .value {
    font-size: 1.8rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
    line-height: 1.15;
    margin-top: 0.2rem;
}
.metric-card .sub {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.15rem;
}
.metric-card.gold  { border-top: 3px solid var(--sun); }
.metric-card.green { border-top: 3px solid var(--good); }
.metric-card.blue  { border-top: 3px solid var(--accent); }
.metric-card.red   { border-top: 3px solid var(--bad); }

/* ── Section headings ── */
.section-head {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--sun);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--panel) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }
.sidebar-logo {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: var(--sun);
    font-weight: 700;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--amber), var(--sun)) !important;
    color: #0D1117 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.3rem !important;
    letter-spacing: 0.04em;
    transition: opacity 0.15s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Selectboxes & sliders ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: var(--deep) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* ── DataFrame ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Info / warning / success boxes ── */
.stAlert { border-radius: 8px !important; }

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div { background: var(--sun) !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--sun) !important;
    border-bottom-color: var(--sun) !important;
}

/* ── Badge ── */
.badge {
    display: inline-block;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
}
.badge-excellent { background: rgba(63,185,80,0.15);  color: var(--good); }
.badge-good      { background: rgba(88,166,255,0.15); color: var(--accent); }
.badge-poor      { background: rgba(248,81,73,0.15);  color: var(--bad); }

/* ── Table ── */
.results-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.results-table th {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid var(--border);
    text-align: left;
}
.results-table td { padding: 0.6rem 0.8rem; border-bottom: 1px solid rgba(48,54,61,0.5); }
.results-table tr:hover td { background: rgba(255,255,255,0.02); }

/* ── Plotly charts dark background ── */
.js-plotly-plot { border-radius: 8px; }

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: var(--panel) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Lazy imports (only load heavy libs when needed) ─────────────────────────
@st.cache_resource
def get_pipeline_modules():
    from src.feature_engineering import (
        chronological_split, engineer_features, get_feature_sets,
    )
    from src.evaluation import evaluate_model
    from src.models import RandomForestModel, XGBoostModel
    return {
        "engineer_features": engineer_features,
        "get_feature_sets": get_feature_sets,
        "chronological_split": chronological_split,
        "evaluate_model": evaluate_model,
        "RandomForestModel": RandomForestModel,
        "XGBoostModel": XGBoostModel,
    }


# ── Plotly theme helper ──────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,34,0.6)",
    font=dict(family="DM Sans, sans-serif", color="#E6EDF3", size=12),
    xaxis=dict(gridcolor="#30363D", zerolinecolor="#30363D"),
    yaxis=dict(gridcolor="#30363D", zerolinecolor="#30363D"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363D"),
)

COUNTRY_COLORS = {"Nigeria": "#F5A623", "Ghana": "#4ECDC4", "Senegal": "#58A6FF"}
MODEL_COLORS   = {"XGBoost": "#3FB950", "Random Forest": "#58A6FF",
                  "LSTM": "#F85149",    "CNN-LSTM": "#BC8CFF"}


# ── Synthetic data generator (demo mode) ────────────────────────────────────
@st.cache_data
def generate_demo_data(country: str, n: int = 750, seed: int = 0) -> pd.DataFrame:
    """Generate realistic synthetic solar data for demo / offline use."""
    rng   = np.random.default_rng(seed + hash(country) % 1000)
    dates = pd.date_range("2021-09-01", periods=n, freq="D")
    doy   = np.arange(1, n + 1)

    # Seasonal GHI signal
    seasonal = 210 + 70 * np.sin(2 * np.pi * (doy - 80) / 365)
    noise    = rng.normal(0, 18, n)
    ghi      = np.clip(seasonal + noise, 60, 340)

    if country == "Senegal":
        ghi *= 1.05
    elif country == "Ghana":
        ghi *= 0.93

    return pd.DataFrame({
        "Date":                   dates,
        "GHI_kWh_m2":            ghi,
        "DNI_kWh_m2":            np.clip(ghi * 0.75 + rng.normal(0, 15, n), 20, 280),
        "DHI_kWh_m2":            np.clip(ghi * 0.30 + rng.normal(0, 8, n),  10, 120),
        "Temperature_C":         25 + 8 * np.sin(2 * np.pi * (doy - 30) / 365) + rng.normal(0, 2, n),
        "Temp_Max_C":            33 + 6 * np.sin(2 * np.pi * (doy - 30) / 365) + rng.normal(0, 1.5, n),
        "Temp_Min_C":            18 + 5 * np.sin(2 * np.pi * (doy - 30) / 365) + rng.normal(0, 1.5, n),
        "Humidity_%":            np.clip(55 - 20 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 7, n), 15, 95),
        "Wind_Speed_m_s":        3.5 + rng.exponential(1.5, n),
        "Wind_Direction_deg":    rng.uniform(0, 360, n),
        "Barometric_Pressure_hPa": 1010 + rng.normal(0, 5, n),
        "Precipitation_mm":      np.clip(rng.exponential(1, n), 0, 40),
        "Country":               country,
    })


# ── Train model (cached by config) ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model(country: str, model_type: str, feature_set: str):
    mods = get_pipeline_modules()
    df_raw = generate_demo_data(country)
    df_eng = mods["engineer_features"](df_raw)
    fs_map = mods["get_feature_sets"](df_eng)
    train, test = mods["chronological_split"](df_eng, test_size=0.20)

    feats = [f for f in fs_map[feature_set] if f in train.columns and f != "GHI_kWh_m2"]
    X_tr  = train[feats].fillna(0)
    y_tr  = train["GHI_kWh_m2"]
    X_te  = test[feats].fillna(0)
    y_te  = test["GHI_kWh_m2"]

    ModelClass = mods["XGBoostModel"] if model_type == "XGBoost" else mods["RandomForestModel"]
    model = ModelClass()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    metrics = mods["evaluate_model"](y_te.values, y_pred, model_type, country, feature_set)

    fi = None
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)

    return {
        "model":   model,
        "train":   train,
        "test":    test,
        "feats":   feats,
        "y_test":  y_te.values,
        "y_pred":  y_pred,
        "metrics": metrics,
        "fi":      fi,
        "df_eng":  df_eng,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          SIDEBAR                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown('<div class="sidebar-logo">☀ SOLAR · FORECAST</div>', unsafe_allow_html=True)
    st.markdown("**West Africa GHI Prediction**")
    st.markdown('<p style="color:#8B949E;font-size:0.78rem;">MSc Dissertation · University of Hull · 2025</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### ⚙️ Configuration")

    country = st.selectbox(
        "Country",
        ["Nigeria", "Ghana", "Senegal"],
        help="Ground station data from World Bank energydata.info",
    )
    model_type = st.selectbox(
        "Model",
        ["XGBoost", "Random Forest"],
        help="XGBoost achieved R²=0.9777 (best). RF is the ensemble partner.",
    )
    feature_set = st.selectbox(
        "Feature Set",
        ["IMPORTANT", "BASE", "FULL"],
        help="IMPORTANT (top-20 by mutual info) consistently outperforms BASE and FULL",
    )

    st.markdown("---")
    run_btn = st.button("▶  Run Pipeline", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📊 Pages")
    page = st.radio(
        "",
        ["🏠 Overview", "📈 Forecast", "🔍 Feature Analysis",
         "🗺️ Geographic", "🔮 Predict New Data"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        '<p style="color:#8B949E;font-size:0.72rem;line-height:1.6;">'
        'Data: World Bank energydata.info<br>'
        'Stations: Bauchi, Kano (NG) · Navrongo, Sunyani (GH) · Ourossogui, Tambacounda (SN)<br>'
        'Period: Sep 2021 – Nov 2023'
        '</p>',
        unsafe_allow_html=True,
    )


# ── Trigger training ─────────────────────────────────────────────────────────
if run_btn or "results" not in st.session_state:
    with st.spinner(f"Training {model_type} on {country} data…"):
        st.session_state["results"] = train_model(country, model_type, feature_set)
        st.session_state["config"]  = (country, model_type, feature_set)

results = st.session_state.get("results")
cfg     = st.session_state.get("config", (country, model_type, feature_set))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          HERO BANNER                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.markdown(f"""
<div class="hero">
  <h1>Solar Radiation Forecasting · West Africa</h1>
  <p>Daily GHI prediction using machine learning — Nigeria · Ghana · Senegal</p>
  <span class="tag">{cfg[0]}</span>
  <span class="tag">{cfg[1]}</span>
  <span class="tag">{cfg[2]} features</span>
  <span class="tag">80/20 chronological split</span>
</div>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       PAGE: OVERVIEW                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if page == "🏠 Overview":
    import plotly.graph_objects as go
    import plotly.express as px

    m = results["metrics"]
    r2_pct = m["R2"] * 100

    # ── KPI cards ──
    rating_cls = "badge-excellent" if m["Rating"] == "Excellent" else \
                 "badge-good"      if m["Rating"] == "Good"      else "badge-poor"

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card gold">
        <div class="label">R² Score</div>
        <div class="value" style="color:#F5A623">{m['R2']:.4f}</div>
        <div class="sub">Variance explained</div>
      </div>
      <div class="metric-card green">
        <div class="label">RMSE</div>
        <div class="value" style="color:#3FB950">{m['RMSE']:.2f}</div>
        <div class="sub">kWh/m² · root mean sq error</div>
      </div>
      <div class="metric-card blue">
        <div class="label">MAE</div>
        <div class="value" style="color:#58A6FF">{m['MAE']:.2f}</div>
        <div class="sub">kWh/m² · mean abs error</div>
      </div>
      <div class="metric-card {'gold' if m['Rating']=='Excellent' else 'red'}">
        <div class="label">MAPE</div>
        <div class="value" style="color:{'#F5A623' if m['MAPE']<6 else '#F85149'}">{m['MAPE']:.1f}%</div>
        <div class="sub"><span class="badge {rating_cls}">{m['Rating']}</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-head">Time-Series Forecast · Test Period</div>', unsafe_allow_html=True)
        y_t   = results["y_test"]
        y_p   = results["y_pred"]
        dates = results["test"]["Date"].values

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=y_t, name="Actual GHI",
            line=dict(color="#58A6FF", width=2), opacity=0.9,
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=y_p, name="Predicted GHI",
            line=dict(color="#F5A623", width=2, dash="dot"), opacity=0.9,
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([dates, dates[::-1]]),
            y=np.concatenate([y_t, y_p[::-1]]),
            fill="toself", fillcolor="rgba(245,166,35,0.06)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
        ))
        fig.update_layout(**PLOTLY_LAYOUT,
                          height=320,
                          title=f"{cfg[1]} · {cfg[0]} · {cfg[2]} features",
                          yaxis_title="GHI (kWh/m²)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-head">Predicted vs Actual</div>', unsafe_allow_html=True)
        from sklearn.metrics import r2_score
        lo, hi = min(y_t.min(), y_p.min()) - 5, max(y_t.max(), y_p.max()) + 5
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=y_t, y=y_p, mode="markers",
            marker=dict(color="#F5A623", size=5, opacity=0.6,
                        line=dict(width=0.5, color="#0D1117")),
            name="Test points",
        ))
        fig2.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi],
            line=dict(color="#8B949E", dash="dash", width=1.5),
            name="Perfect fit", mode="lines",
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=320,
                           xaxis_title="Actual GHI", yaxis_title="Predicted GHI")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Literature benchmark ──
    st.markdown('<div class="section-head">Benchmark vs Literature</div>', unsafe_allow_html=True)
    bench = pd.DataFrame({
        "Study":   ["This study (XGBoost)", "Pinar et al. 2020", "Voyant et al. 2017 (upper)", "Voyant et al. 2017 (typical)", "Yang et al. 2019"],
        "R²":      [0.9777,                  0.88,                0.95,                          0.75,                           0.90],
        "Method":  ["XGBoost",               "Hybrid NN",         "Ensemble",                    "ML average",                   "Deep learning"],
        "Region":  ["West Africa",           "Turkey",            "Various",                      "Various",                      "China"],
    })
    fig3 = px.bar(
        bench, x="Study", y="R²", color="R²",
        color_continuous_scale=["#30363D", "#F5A623"],
        text="R²",
    )
    fig3.update_traces(texttemplate="%{text:.4f}", textposition="outside",
                       textfont_color="#E6EDF3")
    fig3.update_layout(**PLOTLY_LAYOUT, height=300,
                       coloraxis_showscale=False,
                       yaxis_range=[0.6, 1.05])
    st.plotly_chart(fig3, use_container_width=True)

    # ── Data summary ──
    st.markdown('<div class="section-head">Dataset Summary</div>', unsafe_allow_html=True)
    df_eng = results["df_eng"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records",  f"{len(df_eng):,}")
    c2.metric("Training samples", f"{len(results['train']):,}")
    c3.metric("Test samples",    f"{len(results['test']):,}")
    c4.metric("Feature count",   f"{len(results['feats'])}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       PAGE: FORECAST                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

elif page == "📈 Forecast":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.markdown('<div class="section-head">Full Training History + Forecast</div>', unsafe_allow_html=True)

    df_eng  = results["df_eng"]
    train   = results["train"]
    test    = results["test"]
    y_pred  = results["y_pred"]

    fig = make_subplots(rows=2, cols=1,
                        row_heights=[0.7, 0.3],
                        shared_xaxis=True,
                        vertical_spacing=0.06)

    # Training GHI
    fig.add_trace(go.Scatter(
        x=train["Date"], y=train["GHI_kWh_m2"],
        name="Training (actual)", line=dict(color="#30363D", width=1.2),
        fill="tozeroy", fillcolor="rgba(48,54,61,0.3)",
    ), row=1, col=1)

    # Test actual
    fig.add_trace(go.Scatter(
        x=test["Date"], y=test["GHI_kWh_m2"],
        name="Test (actual)", line=dict(color="#58A6FF", width=2),
    ), row=1, col=1)

    # Predicted
    fig.add_trace(go.Scatter(
        x=test["Date"].values[-len(y_pred):], y=y_pred,
        name="Predicted", line=dict(color="#F5A623", width=2, dash="dot"),
    ), row=1, col=1)

    # Residuals
    y_t = test["GHI_kWh_m2"].values[-len(y_pred):]
    residuals = y_t - y_pred
    fig.add_trace(go.Bar(
        x=test["Date"].values[-len(y_pred):], y=residuals,
        name="Residuals",
        marker_color=np.where(residuals >= 0, "#3FB950", "#F85149"),
        opacity=0.7,
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="#8B949E", row=2, col=1)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=520,
        title=f"GHI Forecast — {cfg[0]} | {cfg[1]} | {cfg[2]} features",
    )
    fig.update_yaxes(title_text="GHI (kWh/m²)", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics deep-dive ──
    st.markdown('<div class="section-head">Residual Distribution</div>', unsafe_allow_html=True)
    import plotly.express as px
    col1, col2 = st.columns(2)
    with col1:
        fig_res = px.histogram(
            x=residuals, nbins=30,
            labels={"x": "Residual (kWh/m²)", "y": "Count"},
            color_discrete_sequence=["#F5A623"],
        )
        fig_res.update_layout(**PLOTLY_LAYOUT, height=280,
                              title="Residual histogram",
                              bargap=0.05)
        st.plotly_chart(fig_res, use_container_width=True)

    with col2:
        # QQ-style scatter
        from scipy.stats import probplot
        osm, osr = probplot(residuals, dist="norm")[0]
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                                    marker=dict(color="#58A6FF", size=4),
                                    name="Residuals"))
        lo_q, hi_q = osm.min(), osm.max()
        fig_qq.add_trace(go.Scatter(x=[lo_q, hi_q], y=[lo_q, hi_q],
                                    line=dict(color="#F5A623", dash="dash"),
                                    name="Normal"))
        fig_qq.update_layout(**PLOTLY_LAYOUT, height=280,
                             title="Normal Q-Q plot of residuals",
                             xaxis_title="Theoretical quantiles",
                             yaxis_title="Sample quantiles")
        st.plotly_chart(fig_qq, use_container_width=True)

    # ── Monthly breakdown ──
    st.markdown('<div class="section-head">Monthly Performance</div>', unsafe_allow_html=True)
    test_copy = test.iloc[-len(y_pred):].copy()
    test_copy["Predicted"] = y_pred
    test_copy["AbsError"]  = np.abs(test_copy["GHI_kWh_m2"] - y_pred)
    monthly = test_copy.groupby(test_copy["Date"].dt.month).agg(
        Mean_GHI=("GHI_kWh_m2", "mean"),
        Mean_Pred=("Predicted", "mean"),
        MAE=("AbsError", "mean"),
    ).reset_index()
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly["Month"] = monthly["Date"].map(month_names)

    fig_mon = go.Figure()
    fig_mon.add_trace(go.Bar(x=monthly["Month"], y=monthly["Mean_GHI"],
                             name="Actual mean GHI", marker_color="#58A6FF"))
    fig_mon.add_trace(go.Bar(x=monthly["Month"], y=monthly["Mean_Pred"],
                             name="Predicted mean GHI", marker_color="#F5A623", opacity=0.7))
    fig_mon.update_layout(**PLOTLY_LAYOUT, height=300,
                          barmode="group", yaxis_title="Mean GHI (kWh/m²)")
    st.plotly_chart(fig_mon, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    PAGE: FEATURE ANALYSIS                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

elif page == "🔍 Feature Analysis":
    import plotly.graph_objects as go
    import plotly.express as px

    fi = results["fi"]
    if fi is None:
        st.info("Feature importances not available for this model.")
    else:
        top_n = st.slider("Show top N features", 5, min(30, len(fi)), 15)
        top   = fi.head(top_n)

        st.markdown('<div class="section-head">Feature Importance Ranking</div>', unsafe_allow_html=True)

        # Colour by category
        def cat_color(name: str) -> str:
            n = name.lower()
            if "lag" in n:     return "#F5A623"
            if "roll" in n:    return "#3FB950"
            if any(k in n for k in ("month", "year", "quarter", "season", "sin", "cos", "day")):
                return "#58A6FF"
            return "#BC8CFF"

        colors = [cat_color(n) for n in top.index]
        fig_fi = go.Figure(go.Bar(
            y=top.index[::-1], x=top.values[::-1],
            orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:.4f}" for v in top.values[::-1]],
            textposition="outside",
            textfont=dict(color="#8B949E", size=10),
        ))
        fig_fi.update_layout(**PLOTLY_LAYOUT, height=max(350, top_n * 28),
                             xaxis_title="Feature Importance")
        st.plotly_chart(fig_fi, use_container_width=True)

        # Legend
        st.markdown("""
        <div style="display:flex;gap:1.5rem;font-size:0.78rem;margin-top:-0.5rem;margin-bottom:1rem">
          <span><span style="color:#F5A623">■</span> Lag features</span>
          <span><span style="color:#3FB950">■</span> Rolling statistics</span>
          <span><span style="color:#58A6FF">■</span> Temporal features</span>
          <span><span style="color:#BC8CFF">■</span> Interaction / base features</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Category breakdown ──
        st.markdown('<div class="section-head">Importance by Category</div>', unsafe_allow_html=True)
        cats = {"Lag": 0.0, "Rolling": 0.0, "Temporal": 0.0, "Interaction/Base": 0.0}
        for name, val in fi.items():
            n = name.lower()
            if "lag" in n:
                cats["Lag"] += val
            elif "roll" in n:
                cats["Rolling"] += val
            elif any(k in n for k in ("month", "year", "quarter", "season", "sin", "cos", "day")):
                cats["Temporal"] += val
            else:
                cats["Interaction/Base"] += val

        col1, col2 = st.columns(2)
        with col1:
            fig_pie = go.Figure(go.Pie(
                labels=list(cats.keys()),
                values=list(cats.values()),
                hole=0.45,
                marker_colors=["#F5A623", "#3FB950", "#58A6FF", "#BC8CFF"],
                textfont_color="#E6EDF3",
            ))
            fig_pie.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=True,
                                  title="Importance share by category")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Cumulative importance
            cum   = np.cumsum(fi.values)
            fig_c = go.Figure(go.Scatter(
                x=list(range(1, len(fi)+1)), y=cum,
                fill="tozeroy", fillcolor="rgba(245,166,35,0.12)",
                line=dict(color="#F5A623", width=2),
            ))
            fig_c.add_hline(y=0.8, line_dash="dash", line_color="#3FB950",
                            annotation_text="80%", annotation_font_color="#3FB950")
            fig_c.add_hline(y=0.95, line_dash="dash", line_color="#58A6FF",
                            annotation_text="95%", annotation_font_color="#58A6FF")
            fig_c.update_layout(**PLOTLY_LAYOUT, height=300,
                                title="Cumulative feature importance",
                                xaxis_title="Number of features",
                                yaxis_title="Cumulative importance",
                                yaxis_range=[0, 1.05])
            st.plotly_chart(fig_c, use_container_width=True)

        # ── Raw table ──
        with st.expander("📋 Full feature importance table"):
            fi_df = fi.reset_index()
            fi_df.columns = ["Feature", "Importance"]
            fi_df["Rank"] = range(1, len(fi_df) + 1)
            fi_df["Category"] = fi_df["Feature"].apply(
                lambda n: "Lag" if "lag" in n.lower()
                else "Rolling" if "roll" in n.lower()
                else "Temporal" if any(k in n.lower() for k in ("month","year","quarter","season","sin","cos","day"))
                else "Interaction/Base"
            )
            st.dataframe(fi_df[["Rank","Feature","Category","Importance"]],
                         use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    PAGE: GEOGRAPHIC                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

elif page == "🗺️ Geographic":
    import plotly.graph_objects as go
    import plotly.express as px

    st.markdown('<div class="section-head">Cross-Country Performance Comparison</div>', unsafe_allow_html=True)
    st.info("Results below use the same model configuration on synthetic data for each country. "
            "In production, train on the real World Bank data for each station.", icon="ℹ️")

    mods = get_pipeline_modules()

    @st.cache_data
    def all_country_results(m_type: str, fs: str):
        rows = []
        for c in ["Nigeria", "Ghana", "Senegal"]:
            r = train_model(c, m_type, fs)
            rows.append({**r["metrics"], "TrainSamples": len(r["train"]),
                          "TestSamples": len(r["test"])})
        return pd.DataFrame(rows)

    with st.spinner("Running on all 3 countries…"):
        geo_df = all_country_results(model_type, feature_set)

    # ── R² bar chart ──
    fig_geo = go.Figure()
    fig_geo.add_trace(go.Bar(
        x=geo_df["Country"], y=geo_df["R2"],
        marker_color=[COUNTRY_COLORS[c] for c in geo_df["Country"]],
        text=[f"{v:.4f}" for v in geo_df["R2"]],
        textposition="outside", textfont_color="#E6EDF3",
        name="R² Score",
    ))
    fig_geo.update_layout(**PLOTLY_LAYOUT, height=300,
                          yaxis_range=[0.85, 1.0],
                          yaxis_title="R² Score",
                          title=f"{model_type} · {feature_set} · All countries")
    st.plotly_chart(fig_geo, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_rmse = go.Figure(go.Bar(
            x=geo_df["Country"], y=geo_df["RMSE"],
            marker_color=[COUNTRY_COLORS[c] for c in geo_df["Country"]],
            text=[f"{v:.2f}" for v in geo_df["RMSE"]],
            textposition="outside", textfont_color="#E6EDF3",
        ))
        fig_rmse.update_layout(**PLOTLY_LAYOUT, height=260,
                               yaxis_title="RMSE (kWh/m²)", title="RMSE by country")
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col2:
        fig_mape = go.Figure(go.Bar(
            x=geo_df["Country"], y=geo_df["MAPE"],
            marker_color=[COUNTRY_COLORS[c] for c in geo_df["Country"]],
            text=[f"{v:.1f}%" for v in geo_df["MAPE"]],
            textposition="outside", textfont_color="#E6EDF3",
        ))
        fig_mape.update_layout(**PLOTLY_LAYOUT, height=260,
                               yaxis_title="MAPE (%)", title="MAPE by country")
        st.plotly_chart(fig_mape, use_container_width=True)

    # ── Summary table ──
    st.markdown('<div class="section-head">Results Table</div>', unsafe_allow_html=True)
    disp = geo_df[["Model","Country","FeatureSet","R2","RMSE","MAE","MAPE","Rating"]].copy()
    disp = disp.sort_values("R2", ascending=False).reset_index(drop=True)
    st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── Map ──
    st.markdown('<div class="section-head">Station Locations</div>', unsafe_allow_html=True)
    stations = pd.DataFrame({
        "Station":   ["Bauchi","Kano","Navrongo","Sunyani","Ourossogui","Tambacounda"],
        "Country":   ["Nigeria","Nigeria","Ghana","Ghana","Senegal","Senegal"],
        "lat":       [10.31, 12.00, 10.90, 7.34, 15.65, 13.77],
        "lon":       [9.84,  8.52,  -1.09, -2.33, -13.31, -13.68],
        "R2_approx": [0.977, 0.977, 0.962, 0.962, 0.936, 0.936],
    })
    fig_map = px.scatter_mapbox(
        stations, lat="lat", lon="lon",
        hover_name="Station", hover_data={"Country": True, "R2_approx": ":.4f"},
        color="Country",
        color_discrete_map=COUNTRY_COLORS,
        size=[15]*6, zoom=3, height=380,
        mapbox_style="carto-darkmatter",
    )
    fig_map.update_layout(**PLOTLY_LAYOUT, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_map, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    PAGE: PREDICT NEW DATA                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

elif page == "🔮 Predict New Data":
    import plotly.graph_objects as go

    st.markdown('<div class="section-head">Predict GHI from New Meteorological Data</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📤  Upload CSV", "✏️  Manual Entry"])

    # ── Tab 1: CSV upload ────────────────────────────────────────────────────
    with tab1:
        st.markdown("""
        Upload a CSV with daily meteorological observations.  
        Required columns: `Date`, `GHI_kWh_m2`, `DNI_kWh_m2`, `DHI_kWh_m2`,
        `Temperature_C`, `Temp_Max_C`, `Temp_Min_C`, `Humidity_%`,
        `Wind_Speed_m_s`, `Wind_Direction_deg`, `Barometric_Pressure_hPa`, `Precipitation_mm`
        """)

        uploaded = st.file_uploader("Drop CSV here", type=["csv"])
        if uploaded:
            try:
                new_df = pd.read_csv(uploaded)
                new_df["Date"] = pd.to_datetime(new_df["Date"])
                st.success(f"Loaded {len(new_df)} rows, {len(new_df.columns)} columns")
                st.dataframe(new_df.head(5), use_container_width=True)

                if st.button("Run prediction on uploaded data"):
                    mods   = get_pipeline_modules()
                    df_eng = mods["engineer_features"](new_df)
                    fs_map = mods["get_feature_sets"](df_eng)
                    feats  = [f for f in fs_map[feature_set]
                              if f in df_eng.columns and f != "GHI_kWh_m2"]
                    X      = df_eng[feats].fillna(0)
                    model  = results["model"]
                    preds  = model.predict(X)
                    out    = df_eng[["Date"]].iloc[-len(preds):].copy()
                    out["Predicted_GHI"] = preds
                    if "GHI_kWh_m2" in df_eng.columns:
                        actual = df_eng["GHI_kWh_m2"].values[-len(preds):]
                        out["Actual_GHI"] = actual
                        metrics = mods["evaluate_model"](actual, preds)
                        st.metric("R²", f"{metrics['R2']:.4f}")

                    st.dataframe(out, use_container_width=True)
                    csv = out.to_csv(index=False).encode()
                    st.download_button("⬇ Download predictions", csv,
                                       "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Error processing file: {e}")

    # ── Tab 2: Manual entry ──────────────────────────────────────────────────
    with tab2:
        st.markdown("Enter today's meteorological readings to get a GHI forecast:")

        c1, c2, c3 = st.columns(3)
        with c1:
            dni  = st.number_input("DNI (kWh/m²)",   0.0, 350.0, 180.0, step=1.0)
            dhi  = st.number_input("DHI (kWh/m²)",   0.0, 150.0,  60.0, step=1.0)
            temp = st.number_input("Temperature (°C)", 5.0,  55.0,  28.0, step=0.5)
        with c2:
            hum  = st.number_input("Humidity (%)",    5.0, 100.0,  55.0, step=1.0)
            wind = st.number_input("Wind Speed (m/s)", 0.0,  25.0,   3.5, step=0.1)
            pres = st.number_input("Pressure (hPa)",  950.0, 1050.0, 1010.0, step=0.5)
        with c3:
            wdir  = st.number_input("Wind Direction (°)", 0.0, 360.0, 180.0, step=5.0)
            precip = st.number_input("Precipitation (mm)", 0.0,  80.0,   0.0, step=0.5)
            date  = st.date_input("Date", value=pd.Timestamp.today())

        if st.button("🔮  Predict GHI"):
            # Build a mini warm-up dataset (last 30 days of demo data + today)
            mods     = get_pipeline_modules()
            hist_df  = generate_demo_data(cfg[0], n=80)

            today_row = pd.DataFrame([{
                "Date":                   pd.Timestamp(date),
                "GHI_kWh_m2":            (dni * 0.75 + dhi),   # rough estimate for feature warm-up
                "DNI_kWh_m2":            dni,
                "DHI_kWh_m2":            dhi,
                "Temperature_C":         temp,
                "Temp_Max_C":            temp + 4,
                "Temp_Min_C":            temp - 4,
                "Humidity_%":            hum,
                "Wind_Speed_m_s":        wind,
                "Wind_Direction_deg":    wdir,
                "Barometric_Pressure_hPa": pres,
                "Precipitation_mm":      precip,
                "Country":               cfg[0],
            }])

            combined = pd.concat([hist_df, today_row], ignore_index=True)
            combined = combined.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)

            df_eng  = mods["engineer_features"](combined)
            fs_map  = mods["get_feature_sets"](df_eng)
            feats   = [f for f in fs_map[feature_set]
                       if f in df_eng.columns and f != "GHI_kWh_m2"]
            X_last  = df_eng[feats].fillna(0).iloc[[-1]]
            pred    = results["model"].predict(X_last)[0]

            st.markdown(f"""
            <div class="metric-card gold" style="max-width:320px;margin-top:1rem">
              <div class="label">Predicted GHI for {date}</div>
              <div class="value" style="color:#F5A623">{pred:.1f}</div>
              <div class="sub">kWh/m² · {cfg[1]} · {cfg[0]}</div>
            </div>
            """, unsafe_allow_html=True)

            # Simple gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 340], "tickcolor": "#8B949E"},
                    "bar": {"color": "#F5A623"},
                    "bgcolor": "#161B22",
                    "steps": [
                        {"range": [0,   100], "color": "#21262D"},
                        {"range": [100, 200], "color": "#1C2128"},
                        {"range": [200, 340], "color": "#161B22"},
                    ],
                    "threshold": {"line": {"color": "#3FB950", "width": 4},
                                  "thickness": 0.75, "value": pred},
                },
                number={"suffix": " kWh/m²", "font": {"color": "#E6EDF3"}},
            ))
            fig_g.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E6EDF3",
                height=260,
                margin=dict(l=30, r=30, t=20, b=10),
            )
            st.plotly_chart(fig_g, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1rem;border-top:1px solid #30363D;
            color:#8B949E;font-size:0.72rem;text-align:center;font-family:'Space Mono',monospace;">
  Machine Learning for Daily Solar Radiation Forecasting in West Africa &nbsp;·&nbsp;
  MSc Dissertation &nbsp;·&nbsp; Majid Rasheed &nbsp;·&nbsp; University of Hull &nbsp;·&nbsp; 2025
</div>
""", unsafe_allow_html=True)
