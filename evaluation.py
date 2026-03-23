"""
evaluation.py
-------------
Compute regression metrics (R², RMSE, MAE, MAPE) for any model,
aggregate results into a summary DataFrame, and produce console reports.

Primary functions
-----------------
  evaluate_model()       – returns a dict of metrics for one prediction set
  evaluate_all_models()  – batch-evaluate across countries × feature sets
  build_results_table()  – format results as a tidy DataFrame
  print_summary()        – print a formatted leaderboard to stdout
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.settings import PERFORMANCE_THRESHOLDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-set evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    model_name: str = "",
    country: str = "",
    feature_set: str = "",
) -> dict[str, Any]:
    """
    Compute R², RMSE, MAE, MAPE for one (true, predicted) pair.

    Parameters
    ----------
    y_true, y_pred : array-like
    model_name, country, feature_set : str
        Labels attached to the returned dict.

    Returns
    -------
    dict with keys: Model, Country, FeatureSet, R2, RMSE, MAE, MAPE, Rating
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))

    # MAPE – guard against division by zero
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else np.nan

    thresholds = PERFORMANCE_THRESHOLDS
    if r2 >= thresholds["excellent"]:
        rating = "Excellent"
    elif r2 >= thresholds["good"]:
        rating = "Good"
    elif r2 >= thresholds["acceptable"]:
        rating = "Acceptable"
    else:
        rating = "Poor"

    result = {
        "Model":      model_name,
        "Country":    country.title() if country else "",
        "FeatureSet": feature_set,
        "R2":         round(r2, 4),
        "RMSE":       round(rmse, 4),
        "MAE":        round(mae, 4),
        "MAPE":       round(mape, 2),
        "Rating":     rating,
    }
    logger.info(
        "[%s | %s | %s]  R²=%.4f  RMSE=%.2f  MAE=%.2f  MAPE=%.1f%%  (%s)",
        model_name, country, feature_set, r2, rmse, mae, mape, rating,
    )
    return result


# ---------------------------------------------------------------------------
# Batch evaluation helper
# ---------------------------------------------------------------------------

def evaluate_all_models(
    predictions: dict[tuple[str, str, str], dict],
) -> pd.DataFrame:
    """
    Evaluate a collection of stored predictions.

    Parameters
    ----------
    predictions : dict
        Keys are ``(model_name, country, feature_set)`` tuples.
        Values are dicts with ``"y_test"`` and ``"y_pred"`` arrays.

    Returns
    -------
    pd.DataFrame
        Tidy results table, sorted by R² descending.
    """
    rows = []
    for (model, country, fs), data in predictions.items():
        row = evaluate_model(
            y_true=data["y_test"],
            y_pred=data["y_pred"],
            model_name=model,
            country=country,
            feature_set=fs,
        )
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def build_results_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot results into a compact comparison table (best config per model).

    Returns
    -------
    pd.DataFrame with columns: Model, Best R², Best RMSE, Country, FeatureSet
    """
    idx = results_df.groupby("Model")["R2"].idxmax()
    best = results_df.loc[idx, ["Model", "R2", "RMSE", "MAE", "Country", "FeatureSet", "Rating"]].copy()
    best = best.rename(columns={"R2": "Best R²", "RMSE": "Best RMSE", "MAE": "Best MAE"})
    return best.reset_index(drop=True)


def print_summary(results_df: pd.DataFrame) -> None:
    """Print a formatted leaderboard to stdout."""
    summary = build_results_table(results_df)
    header = f"\n{'='*72}\n{'MODEL PERFORMANCE SUMMARY':^72}\n{'='*72}"
    print(header)
    print(summary.to_string(index=False))
    print("=" * 72)

    best_row = summary.iloc[0]
    print(
        f"\n✓ Best model: {best_row['Model']}  "
        f"(R²={best_row['Best R²']:.4f}, RMSE={best_row['Best RMSE']:.2f})\n"
        f"  Configuration: {best_row['Country']} | {best_row['FeatureSet']}\n"
    )


def save_results(results_df: pd.DataFrame, path: str = "results/results.csv") -> None:
    """Persist the results DataFrame to *path*."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(path, index=False)
    logger.info("Results saved to %s", path)
