"""
app.py  –  Solar Radiation Forecasting Dashboard
All pipeline code is inlined – no src/ imports needed.
Run: streamlit run app.py
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(
    page_title="Solar Forecasting · West Africa",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --sun:#F5A623; --amber:#E8861A; --deep:#0D1117; --panel:#161B22;
    --border:#30363D; --text:#E6EDF3; --muted:#8B949E;
    --good:#3FB950; --warn:#D29922; --bad:#F85149; --accent:#58A6FF;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--deep)!important;color:var(--text)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:1.5rem 2.5rem 2rem!important;max-width:1400px;}
.hero{background:linear-gradient(135deg,#1a1200 0%,#2d1f00 40%,#0D1117 100%);border:1px solid var(--border);border-radius:12px;padding:2rem 2.5rem;margin-bottom:1.5rem;position:relative;overflow:hidden;}
.hero::before{content:'☀';position:absolute;right:2rem;top:-0.5rem;font-size:8rem;opacity:0.06;line-height:1;}
.hero h1{font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;color:var(--sun)!important;margin:0 0 0.3rem!important;letter-spacing:-0.5px;}
.hero p{color:var(--muted);font-size:0.9rem;margin:0!important;}
.hero .tag{display:inline-block;background:rgba(245,166,35,0.12);border:1px solid rgba(245,166,35,0.3);color:var(--sun);border-radius:20px;padding:2px 10px;font-size:0.72rem;font-family:'Space Mono',monospace;margin-right:6px;margin-top:8px;}
.metric-row{display:flex;gap:1rem;margin-bottom:1.5rem;flex-wrap:wrap;}
.metric-card{flex:1;min-width:140px;background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:1.1rem 1.2rem;}
.metric-card .label{font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;font-family:'Space Mono',monospace;}
.metric-card .value{font-size:1.8rem;font-weight:600;font-family:'Space Mono',monospace;line-height:1.15;margin-top:0.2rem;}
.metric-card .sub{font-size:0.75rem;color:var(--muted);margin-top:0.15rem;}
.metric-card.gold{border-top:3px solid var(--sun);}
.metric-card.green{border-top:3px solid var(--good);}
.metric-card.blue{border-top:3px solid var(--accent);}
.metric-card.red{border-top:3px solid var(--bad);}
.section-head{font-family:'Space Mono',monospace;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.12em;color:var(--sun);border-bottom:1px solid var(--border);padding-bottom:0.5rem;margin:1.5rem 0 1rem;}
[data-testid="stSidebar"]{background-color:var(--panel)!important;border-right:1px solid var(--border)!important;}
.sidebar-logo{font-family:'Space Mono',monospace;font-size:1rem;color:var(--sun);font-weight:700;padding-bottom:1rem;border-bottom:1px solid var(--border);margin-bottom:1rem;}
.stButton>button{background:linear-gradient(135deg,var(--amber),var(--sun))!important;color:#0D1117!important;font-family:'Space Mono',monospace!important;font-weight:700!important;font-size:0.8rem!important;border:none!important;border-radius:8px!important;padding:0.55rem 1.3rem!important;width:100%;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# PIPELINE FUNCTIONS (inlined — no src/ imports)
# ============================================================

TARGET = "GHI_kWh_m2"

IMPORTANT_FEATURES = [
    "DNI_kWh_m2", "Month", "DHI_kWh_m2",
    "GHI_kWh_m2_lag1", "DHI_kWh_m2_lag1", "Year_Feature",
    "DHI_kWh_m2_lag7", "Wind_Direction_deg", "DNI_kWh_m2_lag1",
    "GHI_kWh_m2_lag14", "Wind_Speed_m_s", "DNI_kWh_m2_lag30",
    "GHI_kWh_m2_lag7", "DNI_kWh_m2_lag7", "GHI_kWh_m2_lag30",
    "GHI_kWh_m2_roll7_mean", "Temperature_C", "Clearness_Index",
    "DNI_DHI_Ratio", "Humidity_%",
]

BASE_FEATURES = [
    "GHI_kWh_m2", "DNI_kWh_m2", "DHI_kWh_m2",
    "Temperature_C", "Temp_Max_C", "Temp_Min_C",
    "Humidity_%", "Wind_Speed_m_s", "Wind_Direction_deg",
    "Barometric_Pressure_hPa", "Precipitation_mm",
]


def engineer_features(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    doy = df["Date"].dt.dayofyear
    df["Month"]         = df["Date"].dt.month
    df["Year_Feature"]  = df["Date"].dt.year
    df["Quarter"]       = df["Date"].dt.quarter
    df["DayOfWeek"]     = df["Date"].dt.dayofweek
    df["IsWeekend"]     = (df["DayOfWeek"] >= 5).astype(int)
    df["Season"]        = df["Month"].isin({4, 5, 6, 7, 8, 9}).astype(int)
    df["Month_sin"]     = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"]     = np.cos(2 * np.pi * df["Month"] / 12)
    df["DayOfYear_sin"] = np.sin(2 * np.pi * doy / 365)
    df["DayOfYear_cos"] = np.cos(2 * np.pi * doy / 365)
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
    for var in ["GHI_kWh_m2", "DNI_kWh_m2", "DHI_kWh_m2",
                "Temperature_C", "Humidity_%", "Wind_Speed_m_s"]:
        if var not in df.columns:
            continue
        for lag in [1, 7, 14, 30]:
            df[f"{var}_lag{lag}"] = df[var].shift(lag)
    for var in ["GHI_kWh_m2", "DNI_kWh_m2", "Temperature_C"]:
        if var not in df.columns:
            continue
        for win in [7, 14, 30]:
            r = df[var].rolling(win)
            df[f"{var}_roll{win}_mean"] = r.mean()
            df[f"{var}_roll{win}_std"]  = r.std()
            df[f"{var}_roll{win}_min"]  = r.min()
            df[f"{var}_roll{win}_max"]  = r.max()
    if "Temp_Max_C" in df.columns and "Temp_Min_C" in df.columns:
        df["DTR"] = df["Temp_Max_C"] - df["Temp_Min_C"]
    if "Temperature_C" in df.columns:
        df["Temp_Range_7d"] = (df["Temperature_C"].rolling(7).max()
                               - df["Temperature_C"].rolling(7).min())
    if "Humidity_%" in df.columns and "Temperature_C" in df.columns:
        df["Humidity_Temp"] = df["Humidity_%"] * df["Temperature_C"] / 100
    if "Wind_Speed_m_s" in df.columns and "Temperature_C" in df.columns:
        df["Wind_Temp"] = df["Wind_Speed_m_s"] * df["Temperature_C"]
    if "DNI_kWh_m2" in df.columns and "DHI_kWh_m2" in df.columns:
        df["DNI_DHI_Ratio"] = np.where(
            df["DHI_kWh_m2"] > 0, df["DNI_kWh_m2"] / df["DHI_kWh_m2"], 0.0)
        df["GHI_DNI_Ratio"] = np.where(
            df["DNI_kWh_m2"] > 0, df["GHI_kWh_m2"] / df["DNI_kWh_m2"], 0.0)
    extra = 1367.0 * (1 + 0.033 * np.cos(2 * np.pi * doy / 365))
    df["Clearness_Index"] = np.clip(
        np.where(extra > 0, df["GHI_kWh_m2"] / extra, 0.0), 0, 1)
    if "DHI_kWh_m2" in df.columns:
        df["Diffuse_Fraction"] = np.clip(
            np.where(df["GHI_kWh_m2"] > 0,
                     df["DHI_kWh_m2"] / df["GHI_kWh_m2"], 0.0), 0, 1)
    return df.dropna().reset_index(drop=True)


def get_feature_sets(df):
    exclude = {"Date", "Country", TARGET}
    full = [c for c in df.columns if c not in exclude]
    base = [f for f in BASE_FEATURES if f in df.columns and f != TARGET]
    imp  = [f for f in IMPORTANT_FEATURES if f in df.columns]
    return {"BASE": base, "IMPORTANT": imp, "FULL": full}


def chronological_split(df, test_size=0.20):
    cutoff = int(len(df) * (1 - test_size))
    return (df.iloc[:cutoff].reset_index(drop=True),
            df.iloc[cutoff:].reset_index(drop=True))


def evaluate_metrics(y_true, y_pred, model_name="", country="", feature_set=""):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    r2   = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mask = y_true != 0
    mape = (float(np.mean(np.abs((y_true[mask] - y_pred[mask])
                                  / y_true[mask])) * 100)
            if mask.any() else 0.0)
    rating = ("Excellent" if r2 >= 0.95 else
              "Good"      if r2 >= 0.90 else
              "Acceptable" if r2 >= 0.80 else "Poor")
    return {"Model": model_name, "Country": country, "FeatureSet": feature_set,
            "R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4),
            "MAPE": round(mape, 2), "Rating": rating}


# ============================================================
# DEMO DATA
# ============================================================

@st.cache_data
def generate_demo_data(country, n=750):
    seed  = {"Nigeria": 0, "Ghana": 1, "Senegal": 2}.get(country, 0)
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2021-09-01", periods=n, freq="D")
    doy   = np.arange(1, n + 1)
    base  = 210 + 70 * np.sin(2 * np.pi * (doy - 80) / 365)
    ghi   = np.clip(base + rng.normal(0, 18, n), 60, 340)
    if country == "Senegal":
        ghi *= 1.05
    if country == "Ghana":
        ghi *= 0.93
    return pd.DataFrame({
        "Date":                     dates,
        "GHI_kWh_m2":              ghi,
        "DNI_kWh_m2":              np.clip(ghi * 0.75 + rng.normal(0, 15, n), 20, 280),
        "DHI_kWh_m2":              np.clip(ghi * 0.30 + rng.normal(0,  8, n), 10, 120),
        "Temperature_C":           25 + 8*np.sin(2*np.pi*(doy-30)/365) + rng.normal(0, 2, n),
        "Temp_Max_C":              33 + 6*np.sin(2*np.pi*(doy-30)/365) + rng.normal(0, 1.5, n),
        "Temp_Min_C":              18 + 5*np.sin(2*np.pi*(doy-30)/365) + rng.normal(0, 1.5, n),
        "Humidity_%":              np.clip(55 - 20*np.sin(2*np.pi*doy/365)
                                            + rng.normal(0, 7, n), 15, 95),
        "Wind_Speed_m_s":          3.5 + rng.exponential(1.5, n),
        "Wind_Direction_deg":      rng.uniform(0, 360, n),
        "Barometric_Pressure_hPa": 1010 + rng.normal(0, 5, n),
        "Precipitation_mm":        np.clip(rng.exponential(1, n), 0, 40),
        "Country":                 country,
    })


# ============================================================
# MODEL TRAINING
# ============================================================

@st.cache_resource(show_spinner=False)
def train_model(country, model_type, feature_set):
    df_raw  = generate_demo_data(country)
    df_eng  = engineer_features(df_raw)
    fs_map  = get_feature_sets(df_eng)
    train, test = chronological_split(df_eng, test_size=0.20)
    feats   = [f for f in fs_map[feature_set]
               if f in train.columns and f != TARGET]
    X_tr    = train[feats].fillna(0)
    y_tr    = train[TARGET]
    X_te    = test[feats].fillna(0)
    y_te    = test[TARGET]
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    if model_type == "XGBoost":
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=15,
                                      min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X_tr_sc, y_tr)
    y_pred  = model.predict(X_te_sc)
    metrics = evaluate_metrics(y_te.values, y_pred, model_type, country, feature_set)
    fi = None
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_,
                       index=feats).sort_values(ascending=False)
    return {
        "model": model, "scaler": scaler,
        "train": train, "test": test,
        "feats": feats, "y_test": y_te.values, "y_pred": y_pred,
        "metrics": metrics, "fi": fi, "df_eng": df_eng,
    }


# ============================================================
# PLOTLY THEME
# ============================================================

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


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="sidebar-logo">☀ SOLAR · FORECAST</div>',
                unsafe_allow_html=True)
    st.markdown("**West Africa GHI Prediction**")
    st.markdown(
        '<p style="color:#8B949E;font-size:0.78rem;">'
        'MSc Dissertation · University of Hull · 2025</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("#### ⚙️ Configuration")
    country     = st.selectbox("Country",     ["Nigeria", "Ghana", "Senegal"])
    model_type  = st.selectbox("Model",       ["XGBoost", "Random Forest"])
    feature_set = st.selectbox("Feature Set", ["IMPORTANT", "BASE", "FULL"])
    st.caption(
        "ℹ️ LSTM & CNN-LSTM require TensorFlow which is unavailable on "
        "Streamlit Cloud. Their dissertation results (R²≈0.05) are shown "
        "in the Overview page for comparison."
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
        'Period: Sep 2021 – Nov 2023</p>',
        unsafe_allow_html=True,
    )


# ── Trigger training ──────────────────────────────────────────────────────────
if run_btn or "results" not in st.session_state:
    with st.spinner(f"Training {model_type} on {country} data…"):
        st.session_state["results"] = train_model(country, model_type, feature_set)
        st.session_state["config"]  = (country, model_type, feature_set)

results = st.session_state["results"]
cfg     = st.session_state.get("config", (country, model_type, feature_set))


# ── Hero ──────────────────────────────────────────────────────────────────────
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


# ============================================================
# PAGE: OVERVIEW
# ============================================================

if page == "🏠 Overview":
    import plotly.graph_objects as go
    import plotly.express as px

    m = results["metrics"]
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
        <div class="sub">kWh/m²</div>
      </div>
      <div class="metric-card blue">
        <div class="label">MAE</div>
        <div class="value" style="color:#58A6FF">{m['MAE']:.2f}</div>
        <div class="sub">kWh/m²</div>
      </div>
      <div class="metric-card {'gold' if m['MAPE'] < 6 else 'red'}">
        <div class="label">MAPE</div>
        <div class="value" style="color:{'#F5A623' if m['MAPE'] < 6 else '#F85149'}">{m['MAPE']:.1f}%</div>
        <div class="sub">{m['Rating']}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="section-head">Time-Series Forecast · Test Period</div>',
                    unsafe_allow_html=True)
        y_t   = results["y_test"]
        y_p   = results["y_pred"]
        dates = results["test"]["Date"].values
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=y_t, name="Actual GHI",
                                  line=dict(color="#58A6FF", width=2)))
        fig.add_trace(go.Scatter(x=dates, y=y_p, name="Predicted GHI",
                                  line=dict(color="#F5A623", width=2, dash="dot")))
        fig.update_layout(**PLOTLY_LAYOUT, height=320,
                          yaxis_title="GHI (kWh/m²)",
                          title=f"{cfg[1]} · {cfg[0]} · {cfg[2]}")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-head">Predicted vs Actual</div>',
                    unsafe_allow_html=True)
        lo  = float(min(y_t.min(), y_p.min())) - 5
        hi  = float(max(y_t.max(), y_p.max())) + 5
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=y_t, y=y_p, mode="markers",
                                   marker=dict(color="#F5A623", size=5, opacity=0.6)))
        fig2.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi],
                                   line=dict(color="#8B949E", dash="dash", width=1.5),
                                   name="Perfect fit", mode="lines"))
        fig2.update_layout(**PLOTLY_LAYOUT, height=320,
                           xaxis_title="Actual GHI",
                           yaxis_title="Predicted GHI")
        st.plotly_chart(fig2, use_container_width=True)

    # All 4 models — tree-based live + LSTM/CNN-LSTM from dissertation
    st.markdown('<div class="section-head">All Models · Dissertation Results</div>',
                unsafe_allow_html=True)
    all_models = pd.DataFrame({
        "Model":   ["XGBoost", "Random Forest", "LSTM",            "CNN-LSTM"],
        "Best R²": [0.9777,     0.9728,          0.0523,            0.0525],
        "Config":  ["Nigeria IMPORTANT", "Nigeria IMPORTANT",
                    "All configurations", "All configurations"],
        "Status":  ["✅ Live", "✅ Live", "📄 Dissertation only", "📄 Dissertation only"],
    })
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig_all = go.Figure(go.Bar(
            x=all_models["Model"],
            y=all_models["Best R²"],
            marker_color=["#3FB950", "#58A6FF", "#F85149", "#BC8CFF"],
            text=[f"{v:.4f}" for v in all_models["Best R²"]],
            textposition="outside",
            textfont_color="#E6EDF3",
        ))
        fig_all.update_layout(**PLOTLY_LAYOUT, height=320,
                              yaxis_title="Best R² Score",
                              yaxis_range=[0, 1.1],
                              title="All 4 models — dissertation results")
        st.plotly_chart(fig_all, use_container_width=True)
    with col_b:
        st.markdown('<div class="section-head">Summary</div>', unsafe_allow_html=True)
        st.dataframe(all_models[["Model", "Best R²", "Status"]],
                     use_container_width=True, hide_index=True)
        st.caption(
            "LSTM & CNN-LSTM achieved R²≈0.05 due to insufficient training data "
            "(~570 samples vs ~17,000 required). XGBoost is the recommended model."
        )

    st.markdown('<div class="section-head">Benchmark vs Literature</div>',
                unsafe_allow_html=True)
    bench = pd.DataFrame({
        "Study":  ["This study", "Pinar et al. 2020",
                   "Voyant et al. 2017 (upper)", "Voyant et al. 2017 (avg)",
                   "Yang et al. 2019"],
        "R²":     [m["R2"], 0.88, 0.95, 0.75, 0.90],
        "Method": ["XGBoost", "Hybrid NN", "Ensemble", "ML average", "Deep learning"],
    })
    fig3 = px.bar(bench, x="Study", y="R²", color="R²",
                  color_continuous_scale=["#30363D", "#F5A623"], text="R²")
    fig3.update_traces(texttemplate="%{text:.4f}", textposition="outside",
                       textfont_color="#E6EDF3")
    fig3.update_layout(**PLOTLY_LAYOUT, height=300,
                       coloraxis_showscale=False, yaxis_range=[0.6, 1.05])
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-head">Dataset Summary</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records",    f"{len(results['df_eng']):,}")
    c2.metric("Training samples", f"{len(results['train']):,}")
    c3.metric("Test samples",     f"{len(results['test']):,}")
    c4.metric("Feature count",    f"{len(results['feats'])}")


# ============================================================
# PAGE: FORECAST
# ============================================================

elif page == "📈 Forecast":
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    st.markdown('<div class="section-head">Full Training History + Forecast</div>',
                unsafe_allow_html=True)
    train  = results["train"]
    test   = results["test"]
    y_pred = results["y_pred"]
    y_test = results["y_test"]

    fig = make_subplots(rows=2, cols=1, shared_xaxis=True, vertical_spacing=0.06)
    fig.add_trace(
        go.Scatter(x=train["Date"], y=train[TARGET], name="Training",
                   line=dict(color="#30363D", width=1.2),
                   fill="tozeroy", fillcolor="rgba(48,54,61,0.3)"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=test["Date"], y=test[TARGET], name="Test (actual)",
                   line=dict(color="#58A6FF", width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=test["Date"].values[-len(y_pred):], y=y_pred,
                   name="Predicted",
                   line=dict(color="#F5A623", width=2, dash="dot")),
        row=1, col=1,
    )
    residuals = y_test[-len(y_pred):] - y_pred
    fig.add_trace(
        go.Bar(x=test["Date"].values[-len(y_pred):], y=residuals,
               name="Residuals",
               marker_color=np.where(residuals >= 0, "#3FB950", "#F85149"),
               opacity=0.7),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="solid", line_color="#8B949E", row=2, col=1)
    fig.update_layout(**PLOTLY_LAYOUT, height=520,
                      title=f"GHI Forecast — {cfg[0]} | {cfg[1]} | {cfg[2]}")
    fig.update_yaxes(title_text="GHI (kWh/m²)", row=1, col=1)
    fig.update_yaxes(title_text="Residual",      row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-head">Residual Distribution</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig_r = px.histogram(x=residuals, nbins=30,
                             labels={"x": "Residual (kWh/m²)", "y": "Count"},
                             color_discrete_sequence=["#F5A623"])
        fig_r.update_layout(**PLOTLY_LAYOUT, height=280,
                            title="Residual histogram", bargap=0.05)
        st.plotly_chart(fig_r, use_container_width=True)
    with col2:
        from scipy.stats import probplot
        osm, osr = probplot(residuals, dist="norm")[0]
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                                     marker=dict(color="#58A6FF", size=4),
                                     name="Residuals"))
        fig_qq.add_trace(go.Scatter(
            x=[float(osm.min()), float(osm.max())],
            y=[float(osm.min()), float(osm.max())],
            line=dict(color="#F5A623", dash="dash"), name="Normal",
        ))
        fig_qq.update_layout(**PLOTLY_LAYOUT, height=280, title="Normal Q-Q plot",
                             xaxis_title="Theoretical quantiles",
                             yaxis_title="Sample quantiles")
        st.plotly_chart(fig_qq, use_container_width=True)

    st.markdown('<div class="section-head">Monthly Performance</div>',
                unsafe_allow_html=True)
    tc = test.iloc[-len(y_pred):].copy()
    tc["Predicted"] = y_pred
    tc["AbsError"]  = np.abs(tc[TARGET] - y_pred)
    monthly = tc.groupby(tc["Date"].dt.month).agg(
        Mean_GHI=(TARGET, "mean"),
        Mean_Pred=("Predicted", "mean"),
        MAE=("AbsError", "mean"),
    ).reset_index()
    mn = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
          7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly["Month"] = monthly["Date"].map(mn)
    fig_m = go.Figure()
    fig_m.add_trace(go.Bar(x=monthly["Month"], y=monthly["Mean_GHI"],
                            name="Actual", marker_color="#58A6FF"))
    fig_m.add_trace(go.Bar(x=monthly["Month"], y=monthly["Mean_Pred"],
                            name="Predicted", marker_color="#F5A623", opacity=0.7))
    fig_m.update_layout(**PLOTLY_LAYOUT, height=300, barmode="group",
                        yaxis_title="Mean GHI (kWh/m²)")
    st.plotly_chart(fig_m, use_container_width=True)


# ============================================================
# PAGE: FEATURE ANALYSIS
# ============================================================

elif page == "🔍 Feature Analysis":
    import plotly.graph_objects as go

    fi = results["fi"]
    if fi is None:
        st.info("Feature importances not available.")
    else:
        top_n = st.slider("Show top N features", 5, min(30, len(fi)), 15)
        top   = fi.head(top_n)
        st.markdown('<div class="section-head">Feature Importance Ranking</div>',
                    unsafe_allow_html=True)

        def cat_color(name):
            n = name.lower()
            if "lag" in n:
                return "#F5A623"
            if "roll" in n:
                return "#3FB950"
            if any(k in n for k in ("month","year","quarter","season","sin","cos","day")):
                return "#58A6FF"
            return "#BC8CFF"

        colors = [cat_color(n) for n in top.index]
        fig_fi = go.Figure(go.Bar(
            y=top.index[::-1], x=top.values[::-1], orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:.4f}" for v in top.values[::-1]],
            textposition="outside",
            textfont=dict(color="#8B949E", size=10),
        ))
        fig_fi.update_layout(**PLOTLY_LAYOUT, height=max(350, top_n * 28),
                             xaxis_title="Feature Importance")
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("""
        <div style="display:flex;gap:1.5rem;font-size:0.78rem;margin-bottom:1rem">
          <span><span style="color:#F5A623">■</span> Lag</span>
          <span><span style="color:#3FB950">■</span> Rolling</span>
          <span><span style="color:#58A6FF">■</span> Temporal</span>
          <span><span style="color:#BC8CFF">■</span> Interaction/Base</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-head">Importance by Category</div>',
                    unsafe_allow_html=True)
        cats = {"Lag": 0.0, "Rolling": 0.0, "Temporal": 0.0, "Interaction/Base": 0.0}
        for name, val in fi.items():
            n = name.lower()
            if "lag" in n:
                cats["Lag"] += val
            elif "roll" in n:
                cats["Rolling"] += val
            elif any(k in n for k in ("month","year","quarter","season","sin","cos","day")):
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
            fig_pie.update_layout(**PLOTLY_LAYOUT, height=300,
                                  title="Importance by category")
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            cum = np.cumsum(fi.values)
            fig_c = go.Figure(go.Scatter(
                x=list(range(1, len(fi) + 1)), y=cum,
                fill="tozeroy", fillcolor="rgba(245,166,35,0.12)",
                line=dict(color="#F5A623", width=2),
            ))
            fig_c.add_hline(y=0.8,  line_dash="dash", line_color="#3FB950",
                            annotation_text="80%",  annotation_font_color="#3FB950")
            fig_c.add_hline(y=0.95, line_dash="dash", line_color="#58A6FF",
                            annotation_text="95%",  annotation_font_color="#58A6FF")
            fig_c.update_layout(**PLOTLY_LAYOUT, height=300,
                                title="Cumulative importance",
                                xaxis_title="Features",
                                yaxis_title="Cumulative",
                                yaxis_range=[0, 1.05])
            st.plotly_chart(fig_c, use_container_width=True)

        with st.expander("📋 Full feature importance table"):
            fi_df = fi.reset_index()
            fi_df.columns = ["Feature", "Importance"]
            fi_df["Rank"] = range(1, len(fi_df) + 1)
            fi_df["Category"] = fi_df["Feature"].apply(
                lambda n: "Lag" if "lag" in n.lower()
                else "Rolling" if "roll" in n.lower()
                else "Temporal" if any(k in n.lower() for k in
                                       ("month","year","quarter","season","sin","cos","day"))
                else "Interaction/Base"
            )
            st.dataframe(fi_df[["Rank","Feature","Category","Importance"]],
                         use_container_width=True, hide_index=True)


# ============================================================
# PAGE: GEOGRAPHIC
# ============================================================

elif page == "🗺️ Geographic":
    import plotly.graph_objects as go
    import plotly.express as px

    st.markdown('<div class="section-head">Cross-Country Performance Comparison</div>',
                unsafe_allow_html=True)

    @st.cache_data
    def all_country_results(m_type, fs):
        rows = []
        for c in ["Nigeria", "Ghana", "Senegal"]:
            r = train_model(c, m_type, fs)
            rows.append(r["metrics"])
        return pd.DataFrame(rows)

    with st.spinner("Running on all 3 countries…"):
        geo_df = all_country_results(model_type, feature_set)

    fig_geo = go.Figure(go.Bar(
        x=geo_df["Country"], y=geo_df["R2"],
        marker_color=[COUNTRY_COLORS[c] for c in geo_df["Country"]],
        text=[f"{v:.4f}" for v in geo_df["R2"]],
        textposition="outside", textfont_color="#E6EDF3",
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

    st.markdown('<div class="section-head">Results Table</div>', unsafe_allow_html=True)
    st.dataframe(geo_df.sort_values("R2", ascending=False).reset_index(drop=True),
                 use_container_width=True, hide_index=True)

    st.markdown('<div class="section-head">Station Locations</div>',
                unsafe_allow_html=True)
    stations = pd.DataFrame({
        "Station": ["Bauchi","Kano","Navrongo","Sunyani","Ourossogui","Tambacounda"],
        "Country": ["Nigeria","Nigeria","Ghana","Ghana","Senegal","Senegal"],
        "lat":     [10.31, 12.00, 10.90,  7.34, 15.65, 13.77],
        "lon":     [ 9.84,  8.52, -1.09, -2.33,-13.31,-13.68],
    })
    fig_map = px.scatter_mapbox(
        stations, lat="lat", lon="lon", hover_name="Station",
        hover_data={"Country": True}, color="Country",
        color_discrete_map=COUNTRY_COLORS,
        size=[15]*6, zoom=3, height=380, mapbox_style="carto-darkmatter",
    )
    fig_map.update_layout(**PLOTLY_LAYOUT, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_map, use_container_width=True)


# ============================================================
# PAGE: PREDICT NEW DATA
# ============================================================

elif page == "🔮 Predict New Data":
    import plotly.graph_objects as go

    st.markdown('<div class="section-head">Predict GHI from New Data</div>',
                unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📤  Upload CSV", "✏️  Manual Entry"])

    with tab1:
        st.markdown(
            "Upload a CSV with daily meteorological observations "
            "(same columns as the training data)."
        )
        uploaded = st.file_uploader("Drop CSV here", type=["csv"])
        if uploaded:
            try:
                new_df = pd.read_csv(uploaded)
                new_df["Date"] = pd.to_datetime(new_df["Date"])
                st.success(f"Loaded {len(new_df)} rows")
                st.dataframe(new_df.head(), use_container_width=True)
                if st.button("Run prediction"):
                    df_eng = engineer_features(new_df)
                    fs_map = get_feature_sets(df_eng)
                    feats  = [f for f in fs_map[feature_set]
                              if f in df_eng.columns and f != TARGET]
                    X      = results["scaler"].transform(df_eng[feats].fillna(0))
                    preds  = results["model"].predict(X)
                    out    = df_eng[["Date"]].copy()
                    out["Predicted_GHI"] = preds
                    if TARGET in df_eng.columns:
                        out["Actual_GHI"] = df_eng[TARGET].values
                        met = evaluate_metrics(df_eng[TARGET].values, preds)
                        st.metric("R²", f"{met['R2']:.4f}")
                    st.dataframe(out, use_container_width=True)
                    st.download_button(
                        "⬇ Download predictions",
                        out.to_csv(index=False).encode(),
                        "predictions.csv", "text/csv",
                    )
            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        st.markdown("Enter today's meteorological readings:")
        c1, c2, c3 = st.columns(3)
        with c1:
            dni    = st.number_input("DNI (kWh/m²)",      0.0, 350.0, 180.0)
            dhi    = st.number_input("DHI (kWh/m²)",      0.0, 150.0,  60.0)
            temp   = st.number_input("Temperature (°C)",   5.0,  55.0,  28.0)
        with c2:
            hum    = st.number_input("Humidity (%)",       5.0, 100.0,  55.0)
            wind   = st.number_input("Wind Speed (m/s)",   0.0,  25.0,   3.5)
            pres   = st.number_input("Pressure (hPa)",   950.0,1050.0,1010.0)
        with c3:
            wdir   = st.number_input("Wind Direction (°)", 0.0, 360.0, 180.0)
            precip = st.number_input("Precipitation (mm)", 0.0,  80.0,   0.0)
            date   = st.date_input("Date", value=pd.Timestamp.today())

        if st.button("🔮  Predict GHI"):
            hist = generate_demo_data(cfg[0], n=80)
            today_row = pd.DataFrame([{
                "Date":                     pd.Timestamp(date),
                "GHI_kWh_m2":              dni * 0.75 + dhi,
                "DNI_kWh_m2":              dni,
                "DHI_kWh_m2":              dhi,
                "Temperature_C":           temp,
                "Temp_Max_C":              temp + 4,
                "Temp_Min_C":              temp - 4,
                "Humidity_%":              hum,
                "Wind_Speed_m_s":          wind,
                "Wind_Direction_deg":      wdir,
                "Barometric_Pressure_hPa": pres,
                "Precipitation_mm":        precip,
                "Country":                 cfg[0],
            }])
            combined = (pd.concat([hist, today_row], ignore_index=True)
                        .drop_duplicates("Date")
                        .sort_values("Date")
                        .reset_index(drop=True))
            df_eng  = engineer_features(combined)
            fs_map  = get_feature_sets(df_eng)
            feats   = [f for f in fs_map[feature_set]
                       if f in df_eng.columns and f != TARGET]
            X_last  = results["scaler"].transform(
                df_eng[feats].fillna(0).iloc[[-1]])
            pred    = float(results["model"].predict(X_last)[0])

            st.markdown(f"""
            <div class="metric-card gold" style="max-width:320px;margin-top:1rem">
              <div class="label">Predicted GHI for {date}</div>
              <div class="value" style="color:#F5A623">{pred:.1f}</div>
              <div class="sub">kWh/m² · {cfg[1]} · {cfg[0]}</div>
            </div>
            """, unsafe_allow_html=True)

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                gauge={
                    "axis":    {"range": [0, 340], "tickcolor": "#8B949E"},
                    "bar":     {"color": "#F5A623"},
                    "bgcolor": "#161B22",
                    "steps":   [{"range": [0,   100], "color": "#21262D"},
                                 {"range": [100, 200], "color": "#1C2128"},
                                 {"range": [200, 340], "color": "#161B22"}],
                },
                number={"suffix": " kWh/m²", "font": {"color": "#E6EDF3"}},
            ))
            fig_g.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#E6EDF3",
                height=260, margin=dict(l=30, r=30, t=20, b=10),
            )
            st.plotly_chart(fig_g, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1rem;border-top:1px solid #30363D;
            color:#8B949E;font-size:0.72rem;text-align:center;
            font-family:'Space Mono',monospace;">
  Machine Learning for Daily Solar Radiation Forecasting in West Africa &nbsp;·&nbsp;
  MSc Dissertation &nbsp;·&nbsp; Majid Rasheed &nbsp;·&nbsp;
  University of Hull &nbsp;·&nbsp; 2025
</div>
""", unsafe_allow_html=True)
