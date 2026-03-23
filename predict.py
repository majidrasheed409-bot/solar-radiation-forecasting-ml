"""
predict.py
----------
Run inference with a saved model on new meteorological data.

Usage (CLI)
-----------
    # Predict from a CSV file of new observations
    python scripts/predict.py \
        --model-path saved_models/xgboost_nigeria_IMPORTANT.pkl \
        --input data_cache/data_Nigeria_2021_2023.csv \
        --country nigeria \
        --output results/predictions.csv

    # Quick smoke-test with the most recent 60 days of cached data
    python scripts/predict.py --smoke-test --country nigeria

The script:
  1. Loads a saved model (.pkl)
  2. Applies the same feature engineering pipeline
  3. Selects the correct feature set (read from the model's metadata)
  4. Returns a DataFrame with Date, Actual_GHI (if available), Predicted_GHI
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_ingestion import download_country_data
from src.evaluation import evaluate_model
from src.feature_engineering import engineer_features, get_feature_sets
from src.models import (
    CNNLSTMModel,
    LSTMModel,
    RandomForestModel,
    XGBoostModel,
    _BaseModel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------

def predict(
    model_path: str | Path,
    input_df: pd.DataFrame,
    feature_set: str = "IMPORTANT",
    country: str = "unknown",
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Apply a saved model to *input_df* and return a results DataFrame.

    Parameters
    ----------
    model_path : str | Path
        Path to a ``.pkl`` file saved by ``model.save()``.
    input_df : pd.DataFrame
        Raw daily solar data (same schema as data_ingestion output).
        Must include a ``Date`` column and base meteorological variables.
    feature_set : str
        ``"BASE"``, ``"IMPORTANT"``, or ``"FULL"``.
    country : str
        Label used in logging and output columns.
    output_path : str | Path | None
        If provided, save the output DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame
        Columns: Date, Predicted_GHI, [Actual_GHI, R2, RMSE if target present]
    """
    # 1. Load model
    model: _BaseModel = _BaseModel.load(model_path)
    logger.info("Loaded %s from %s", model.name, model_path)

    # 2. Feature engineering
    df_eng = engineer_features(input_df)

    # 3. Select feature columns
    fs_map   = get_feature_sets(df_eng)
    features = [f for f in fs_map.get(feature_set, fs_map["IMPORTANT"])
                if f in df_eng.columns and f != "GHI_kWh_m2"]
    logger.info("Using %d features from %s set", len(features), feature_set)

    X = df_eng[features].fillna(0)

    # 4. Predict
    y_pred = model.predict(X)

    # Align output with dates (neural models skip first `lookback` rows)
    df_out = df_eng[["Date"]].copy().reset_index(drop=True)
    offset = len(df_out) - len(y_pred)
    df_out = df_out.iloc[offset:].reset_index(drop=True)
    df_out["Predicted_GHI_kWh_m2"] = y_pred

    # 5. Attach actual values if available
    if "GHI_kWh_m2" in df_eng.columns:
        y_true = df_eng["GHI_kWh_m2"].values[offset:]
        df_out["Actual_GHI_kWh_m2"] = y_true
        metrics = evaluate_model(y_true, y_pred, model.name, country, feature_set)
        logger.info(
            "Evaluation on provided data → R²=%.4f  RMSE=%.2f  MAE=%.2f",
            metrics["R2"], metrics["RMSE"], metrics["MAE"],
        )
        df_out["Country"] = country.title()

    # 6. Save
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(output_path, index=False)
        logger.info("Predictions saved to %s", output_path)

    return df_out


# ---------------------------------------------------------------------------
# Smoke test helper
# ---------------------------------------------------------------------------

def smoke_test(country: str = "nigeria", n_days: int = 60) -> None:
    """
    Quick validation: load cached data, predict with best saved model, print summary.
    """
    model_path = Path("saved_models") / f"xgboost_{country}_IMPORTANT.pkl"
    if not model_path.exists():
        logger.error("Saved model not found at %s  – run train.py first.", model_path)
        return

    df = download_country_data(country, use_cache=True)
    if df is None:
        logger.error("Could not load data for %s", country)
        return

    df_recent = df.tail(n_days + 60)   # +60 for lag/rolling warm-up
    result_df = predict(model_path, df_recent, feature_set="IMPORTANT", country=country)

    print("\n" + "=" * 60)
    print(f"SMOKE TEST  –  {country.upper()}  (last {n_days} days)")
    print("=" * 60)
    print(result_df.tail(10).to_string(index=False))
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference with a saved solar forecasting model")
    p.add_argument("--model-path",   type=str, help="Path to saved .pkl model")
    p.add_argument("--input",        type=str, help="Path to input CSV (daily meteorological data)")
    p.add_argument("--country",      type=str, default="nigeria",
                   choices=["nigeria", "ghana", "senegal"],
                   help="Country label for logging")
    p.add_argument("--feature-set",  type=str, default="IMPORTANT",
                   choices=["BASE", "IMPORTANT", "FULL"])
    p.add_argument("--output",       type=str, default="results/predictions.csv")
    p.add_argument("--smoke-test",   action="store_true",
                   help="Run a quick validation with cached data (ignores --input)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.smoke_test:
        smoke_test(country=args.country)
    else:
        if not args.model_path or not args.input:
            print("ERROR: --model-path and --input are required unless --smoke-test is set.")
            sys.exit(1)

        input_df = pd.read_csv(args.input)
        if "Date" in input_df.columns:
            input_df["Date"] = pd.to_datetime(input_df["Date"])

        predict(
            model_path=args.model_path,
            input_df=input_df,
            feature_set=args.feature_set,
            country=args.country,
            output_path=args.output,
        )
