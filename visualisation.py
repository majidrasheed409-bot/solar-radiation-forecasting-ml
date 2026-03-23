"""
visualisation.py
----------------
Publication-quality plots reproduced from the dissertation (Figures 1–9).

Functions
---------
  plot_model_comparison()      – Figure 1  bar chart R² across models/countries
  plot_feature_set_comparison()– Figure 2  feature set effect per model
  plot_lstm_training()         – Figure 3  LSTM loss curves
  plot_feature_importance()    – Figure 4  XGBoost top-15 feature importances
  plot_predictions_scatter()   – Figure 5  predicted vs actual scatter
  plot_geographic_performance()– Figure 6  country-level performance gradient
  plot_efficiency()            – Figure 7  performance vs training time bubble
  plot_timeseries_forecast()   – Figure 8  time-series overlay
  plot_feature_pipeline()      – Figure 9  engineering pipeline summary
  save_all_figures()           – convenience wrapper
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.settings import FIGURES_DIR

logger = logging.getLogger(__name__)
plt.style.use("seaborn-v0_8-darkgrid")

_COUNTRY_COLORS = {"Nigeria": "#FF6B6B", "Ghana": "#4ECDC4", "Senegal": "#45B7D1"}
_MODEL_COLORS   = {"XGBoost": "#2ecc71", "Random Forest": "#3498db",
                   "LSTM": "#e74c3c", "CNN-LSTM": "#9b59b6"}


def _save(fig: plt.Figure, filename: str, out_dir: str = FIGURES_DIR) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(out_dir) / filename
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", fp)


# ---------------------------------------------------------------------------
# Figure 1 – Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(results_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Bar chart of R² for all model × country combinations."""
    models   = ["XGBoost", "Random Forest", "LSTM", "CNN-LSTM"]
    countries = ["Nigeria", "Ghana", "Senegal"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x    = np.arange(len(models))
    w    = 0.25
    offs = np.linspace(-(len(countries) - 1) * w / 2, (len(countries) - 1) * w / 2, len(countries))

    for off, country in zip(offs, countries):
        r2_vals = []
        for model in models:
            rows = results_df[
                (results_df["Model"] == model) &
                (results_df["Country"].str.lower() == country.lower())
            ]
            r2_vals.append(rows["R2"].max() if not rows.empty else 0.0)
        ax.bar(x + off, r2_vals, width=w, label=country, color=_COUNTRY_COLORS[country], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Best R² Score", fontsize=12)
    ax.set_title("Model Performance Comparison Across Countries", fontsize=14, fontweight="bold")
    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.6, label="R²=0.90 threshold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    if save:
        _save(fig, "figure1_model_comparison.png")
    return fig


# ---------------------------------------------------------------------------
# Figure 2 – Feature set comparison
# ---------------------------------------------------------------------------

def plot_feature_set_comparison(results_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """R² by feature set for tree-based models on Nigeria."""
    tree_models = ["XGBoost", "Random Forest"]
    feature_sets = ["BASE", "IMPORTANT", "FULL"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x   = np.arange(len(feature_sets))
    w   = 0.35

    for i, model in enumerate(tree_models):
        r2_vals = []
        for fs in feature_sets:
            rows = results_df[
                (results_df["Model"] == model) &
                (results_df["Country"].str.lower() == "nigeria") &
                (results_df["FeatureSet"] == fs)
            ]
            r2_vals.append(rows["R2"].max() if not rows.empty else 0.0)
        offset = (i - 0.5) * w
        ax.bar(x + offset, r2_vals, width=w, label=model,
               color=list(_MODEL_COLORS.values())[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{fs}\n({11 if fs=='BASE' else 20 if fs=='IMPORTANT' else '81'} features)"
                        for fs in feature_sets])
    ax.set_xlabel("Feature Set", fontsize=12)
    ax.set_ylabel("R² Score (Nigeria)", fontsize=12)
    ax.set_title("Feature Set Performance Comparison – Nigeria", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_ylim(0.85, 1.00)
    fig.tight_layout()
    if save:
        _save(fig, "figure2_feature_set_comparison.png")
    return fig


# ---------------------------------------------------------------------------
# Figure 3 – LSTM training curves
# ---------------------------------------------------------------------------

def plot_lstm_training(history_dict: dict, save: bool = True) -> plt.Figure:
    """Plot training vs validation loss from a Keras history object dict."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    train_loss = history_dict.get("loss", [])
    val_loss   = history_dict.get("val_loss", [])
    epochs     = range(1, len(train_loss) + 1)

    axes[0].plot(epochs, train_loss, label="Training Loss", color="#3498db", linewidth=2)
    axes[0].plot(epochs, val_loss,   label="Validation Loss", color="#e74c3c", linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("LSTM Training Curves (Nigeria)", fontsize=13, fontweight="bold")
    axes[0].legend()

    # R² scatter per experiment (static from dissertation results)
    r2_vals  = [0.0523, 0.0521, 0.0518, 0.0531, 0.0525, 0.0527]
    exps     = range(len(r2_vals))
    axes[1].bar(exps, r2_vals, color="#9b59b6", alpha=0.7)
    axes[1].axhline(np.mean(r2_vals), color="red", linestyle="--", label=f"Mean R²={np.mean(r2_vals):.3f}")
    axes[1].set_xlabel("Experiment"); axes[1].set_ylabel("R² Score")
    axes[1].set_title("LSTM R² Across Configurations", fontsize=13, fontweight="bold")
    axes[1].legend()

    fig.suptitle("LSTM Performance – Convergence Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save:
        _save(fig, "figure3_lstm_training.png")
    return fig


# ---------------------------------------------------------------------------
# Figure 4 – Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 15,
    model_name: str = "XGBoost",
    country: str = "Nigeria",
    save: bool = True,
) -> plt.Figure:
    """Horizontal bar chart of top-N feature importances."""
    idx     = np.argsort(importances)[-top_n:]
    names   = np.array(feature_names)[idx]
    vals    = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(names, vals, color="#2ecc71", alpha=0.85)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances: {model_name} – {country}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save:
        _save(fig, "figure4_feature_importance.png")
    return fig


# ---------------------------------------------------------------------------
# Figure 5 – Predicted vs actual scatter
# ---------------------------------------------------------------------------

def plot_predictions_scatter(
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
    country: str = "Nigeria",
    save: bool = True,
) -> plt.Figure:
    """Scatter of predicted vs actual for multiple models side-by-side."""
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    from sklearn.metrics import r2_score

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        # Align lengths (LSTM may have shorter output due to lookback)
        y_t = y_test[-len(y_pred):]
        r2  = r2_score(y_t, y_pred)
        ax.scatter(y_t, y_pred, alpha=0.5, s=15, color=_MODEL_COLORS.get(name, "#34495e"))
        lo  = min(y_t.min(), y_pred.min())
        hi  = max(y_t.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5)
        ax.set_xlabel("Actual GHI (kWh/m²)"); ax.set_ylabel("Predicted GHI (kWh/m²)")
        ax.set_title(f"{name}\n(R²={r2:.4f})", fontsize=11, fontweight="bold")

    fig.suptitle(f"Predicted vs Actual GHI – {country}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save:
        _save(fig, "figure5_predictions_scatter.png")
    return fig


# ---------------------------------------------------------------------------
# Figure 6 – Geographic performance
# ---------------------------------------------------------------------------

def plot_geographic_performance(results_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Line chart of best R² and RMSE across countries."""
    countries = ["Nigeria", "Ghana", "Senegal"]
    tree_models = ["XGBoost", "Random Forest"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for model in tree_models:
        r2_vals, rmse_vals = [], []
        for c in countries:
            rows = results_df[
                (results_df["Model"] == model) &
                (results_df["Country"].str.lower() == c.lower())
            ]
            r2_vals.append(rows["R2"].max()   if not rows.empty else np.nan)
            rmse_vals.append(rows["RMSE"].min() if not rows.empty else np.nan)
        col = _MODEL_COLORS[model]
        axes[0].plot(countries, r2_vals,   "o-", color=col, label=model, linewidth=2, markersize=8)
        axes[1].plot(countries, rmse_vals, "s-", color=col, label=model, linewidth=2, markersize=8)

    axes[0].set_ylabel("Best R² Score", fontsize=12)
    axes[0].set_title("Geographic Performance (R²)", fontsize=13, fontweight="bold")
    axes[0].legend()

    axes[1].set_ylabel("Best RMSE (kWh/m²)", fontsize=12)
    axes[1].set_title("Geographic Performance (RMSE)", fontsize=13, fontweight="bold")
    axes[1].legend()

    fig.suptitle("Performance Gradient Across Nigeria, Ghana, and Senegal",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save:
        _save(fig, "figure6_geographic_performance.png")
    return fig


# ---------------------------------------------------------------------------
# Figure 7 – Computational efficiency
# ---------------------------------------------------------------------------

def plot_efficiency(results_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Bubble chart: R² vs relative training time."""
    models      = ["Random Forest", "XGBoost", "LSTM", "CNN-LSTM"]
    train_times = [1.0, 1.2, 5.0, 6.0]

    r2_vals = []
    for model in models:
        rows = results_df[results_df["Model"] == model]
        r2_vals.append(rows["R2"].max() if not rows.empty else 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [_MODEL_COLORS[m] for m in models]
    scatter = ax.scatter(train_times, r2_vals, s=300, c=colors, alpha=0.8, zorder=5)

    for i, model in enumerate(models):
        ax.annotate(model, (train_times[i], r2_vals[i]),
                    textcoords="offset points", xytext=(0, 12), ha="center", fontsize=10)

    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.5, label="R²=0.90")
    ax.set_xlabel("Relative Training Time", fontsize=12)
    ax.set_ylabel("Best R² Score", fontsize=12)
    ax.set_title("Model Efficiency: Performance vs Training Time", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    if save:
        _save(fig, "figure7_efficiency.png")
    return fig


# ---------------------------------------------------------------------------
# Figure 8 – Time-series forecast
# ---------------------------------------------------------------------------

def plot_timeseries_forecast(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "XGBoost",
    country: str = "Nigeria",
    n_days: int = 60,
    save: bool = True,
) -> plt.Figure:
    """Overlay predicted vs actual GHI over *n_days* of the test period."""
    from sklearn.metrics import r2_score

    n = min(n_days, len(y_pred), len(y_test))
    y_t = y_test[-len(y_pred):][:n]
    y_p = y_pred[:n]
    days = np.arange(n)

    r2 = r2_score(y_t, y_p)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(days, y_t, "b-",  label="Actual",    alpha=0.8, linewidth=2)
    ax.plot(days, y_p, "r--", label="Predicted", alpha=0.8, linewidth=2)
    ax.fill_between(days, y_t, y_p, alpha=0.15, color="gray")

    ax.set_xlabel("Day (Test Period)", fontsize=12)
    ax.set_ylabel("GHI (kWh/m²)", fontsize=12)
    ax.set_title(f"Time-Series Forecast – {model_name} | {country}  (R²={r2:.4f})",
                 fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    if save:
        _save(fig, "figure8_timeseries_forecast.png")
    return fig


# ---------------------------------------------------------------------------
# Figure 9 – Feature engineering pipeline
# ---------------------------------------------------------------------------

def plot_feature_pipeline(
    importances_dict: Optional[dict] = None,
    save: bool = True,
) -> plt.Figure:
    """Four-panel feature engineering dashboard."""
    categories = ["Temporal", "Lag", "Rolling", "Interaction"]
    counts     = [12, 24, 36, 9]
    colors     = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Feature Engineering Pipeline", fontsize=15, fontweight="bold")

    # Top-left: BASE pie
    axes[0, 0].pie([11], labels=["BASE\n(11 features)"], colors=["#3498db"],
                   textprops={"fontsize": 14})
    axes[0, 0].set_title("BASE: Original Features")

    # Top-right: FULL bar
    axes[0, 1].bar(categories, counts, color=colors, alpha=0.85)
    axes[0, 1].set_ylabel("Number of Features")
    axes[0, 1].set_title(f"FULL: {sum(counts)} Engineered Features by Category")

    # Bottom-left: importance pie
    imp_vals = [0.12, 0.50, 0.20, 0.18]   # defaults from dissertation §3.2
    if importances_dict:
        cat_imp = {"Temporal": 0.0, "Lag": 0.0, "Rolling": 0.0, "Interaction": 0.0}
        for fi in importances_dict.values():
            for feat, imp in zip(fi["features"], fi["importances"]):
                fl = feat.lower()
                if "lag" in fl:
                    cat_imp["Lag"] += imp
                elif any(kw in fl for kw in ("roll", "mean", "std")):
                    cat_imp["Rolling"] += imp
                elif any(kw in fl for kw in ("month", "year", "quarter", "sin", "cos", "season")):
                    cat_imp["Temporal"] += imp
                else:
                    cat_imp["Interaction"] += imp
        total = sum(cat_imp.values())
        if total > 0:
            imp_vals = [cat_imp[c] / total for c in categories]

    axes[1, 0].pie(imp_vals, labels=categories, autopct="%1.0f%%",
                   colors=colors, startangle=90)
    axes[1, 0].set_title("Feature Importance by Category (XGBoost)")

    # Bottom-right: IMPORTANT composition
    imp_comp = [8, 5, 4, 3]
    axes[1, 1].barh(categories, imp_comp, color=colors, alpha=0.85)
    axes[1, 1].set_xlabel("Number of Features")
    axes[1, 1].set_title("IMPORTANT Set (top-20) Composition")

    fig.tight_layout()
    if save:
        _save(fig, "figure9_feature_pipeline.png")
    return fig


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def save_all_figures(
    results_df: pd.DataFrame,
    predictions: dict,
    feature_importances: Optional[dict] = None,
    lstm_history: Optional[dict] = None,
) -> None:
    """Generate and save all nine dissertation figures."""
    logger.info("Generating all figures …")

    plot_model_comparison(results_df)
    plot_feature_set_comparison(results_df)

    if lstm_history:
        plot_lstm_training(lstm_history)

    if feature_importances:
        for (model, country), fi in feature_importances.items():
            if model == "XGBoost" and country.lower() == "nigeria":
                plot_feature_importance(fi["features"], fi["importances"],
                                        model_name=model, country=country)
                break

    # Scatter for Nigeria XGBoost + RF
    ng_test = None
    scatter_preds = {}
    for (model, country, fs), data in predictions.items():
        if country.lower() == "nigeria" and fs == "IMPORTANT" and model in ("XGBoost", "Random Forest"):
            scatter_preds[model] = data["y_pred"]
            ng_test = data["y_test"]
    if ng_test is not None:
        plot_predictions_scatter(ng_test, scatter_preds, country="Nigeria")

    plot_geographic_performance(results_df)
    plot_efficiency(results_df)

    # Time-series for best model
    best = results_df.iloc[0]
    key = (best["Model"], best["Country"].lower(), best["FeatureSet"])
    if key in predictions:
        d = predictions[key]
        plot_timeseries_forecast(d["y_test"], d["y_pred"],
                                 model_name=best["Model"], country=best["Country"])

    plot_feature_pipeline(feature_importances)
    logger.info("All figures saved to %s/", FIGURES_DIR)
