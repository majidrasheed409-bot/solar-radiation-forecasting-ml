"""
train.py
--------
End-to-end training pipeline for the West Africa solar forecasting system.

Usage (CLI)
-----------
    python scripts/train.py                        # all models, all countries
    python scripts/train.py --models xgboost rf    # tree-based only
    python scripts/train.py --countries nigeria     # single country
    python scripts/train.py --feature-sets IMPORTANT FULL
    python scripts/train.py --no-cache             # re-download raw data

The script:
  1. Loads / downloads data (energydata.info API or cache)
  2. Engineers 81 features
  3. Builds chronological train/test splits
  4. Trains each selected model × country × feature-set combination
  5. Evaluates and saves results to results/results.csv
  6. Generates all nine dissertation figures
  7. Saves fitted models to saved_models/
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Make src/ importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    COUNTRIES,
    MODEL_SAVE_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_SIZE,
)
from src.data_ingestion import load_all_countries
from src.evaluation import evaluate_model, print_summary, save_results
from src.feature_engineering import (
    chronological_split,
    engineer_features,
    get_feature_sets,
)
from src.models import CNNLSTMModel, LSTMModel, RandomForestModel, XGBoostModel
from src.visualisation import save_all_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "rf":       RandomForestModel,
    "xgboost":  XGBoostModel,
    "lstm":     LSTMModel,
    "cnn_lstm": CNNLSTMModel,
}

MODEL_DISPLAY_NAMES = {
    "rf":       "Random Forest",
    "xgboost":  "XGBoost",
    "lstm":     "LSTM",
    "cnn_lstm": "CNN-LSTM",
}


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def run_experiment(
    countries: list[str] | None = None,
    model_keys: list[str] | None = None,
    feature_set_names: list[str] | None = None,
    use_cache: bool = True,
    save_models: bool = True,
    generate_figures: bool = True,
) -> pd.DataFrame:
    """
    Run the full experiment and return a results DataFrame.

    Parameters
    ----------
    countries : list[str] | None
        Subset of countries to include.  Defaults to all three.
    model_keys : list[str] | None
        Model identifiers (``"rf"``, ``"xgboost"``, ``"lstm"``, ``"cnn_lstm"``).
        Defaults to all four.
    feature_set_names : list[str] | None
        Which feature sets to evaluate (``"BASE"``, ``"IMPORTANT"``, ``"FULL"``).
        Defaults to all three.
    use_cache : bool
        Use local CSV cache if available.
    save_models : bool
        Persist fitted models to ``saved_models/``.
    generate_figures : bool
        Produce all nine dissertation figures.

    Returns
    -------
    pd.DataFrame
        Tidy results table.
    """
    countries        = countries or COUNTRIES
    model_keys       = model_keys or list(MODEL_REGISTRY.keys())
    feature_set_names = feature_set_names or ["BASE", "IMPORTANT", "FULL"]

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Loading solar radiation data")
    logger.info("=" * 60)
    solar_data = load_all_countries(use_cache=use_cache)
    if not solar_data:
        raise RuntimeError("No data loaded.  Check internet access or cache files.")

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Engineering features (81 variables)")
    logger.info("=" * 60)
    engineered = {}
    for country in countries:
        if country not in solar_data:
            logger.warning("No data for %s – skipping.", country)
            continue
        engineered[country] = engineer_features(solar_data[country])
        logger.info("%s: %d rows after engineering", country, len(engineered[country]))

    # ------------------------------------------------------------------
    # 3. Train / test split
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: Chronological 80/20 train-test split")
    logger.info("=" * 60)
    train_data, test_data = {}, {}
    feature_sets_by_country = {}
    for country, df in engineered.items():
        train_data[country], test_data[country] = chronological_split(df, TEST_SIZE)
        feature_sets_by_country[country] = get_feature_sets(df)

    # ------------------------------------------------------------------
    # 4. Model training
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4: Training models")
    logger.info("=" * 60)

    all_results:      list[dict]                             = []
    predictions:      dict[tuple, dict]                      = {}
    feature_importances: dict[tuple, dict]                   = {}
    lstm_history:     Optional[dict]                         = None

    total = len(model_keys) * len(countries) * len(feature_set_names)
    done  = 0

    for mk in model_keys:
        model_name = MODEL_DISPLAY_NAMES[mk]

        for country in countries:
            if country not in train_data:
                continue
            train_df = train_data[country]
            test_df  = test_data[country]
            fs_map   = feature_sets_by_country[country]

            for fs_name in feature_set_names:
                if fs_name not in fs_map:
                    logger.warning("Feature set %s not available – skipping.", fs_name)
                    continue

                features = [f for f in fs_map[fs_name]
                            if f in train_df.columns and f != "GHI_kWh_m2"]
                if not features:
                    logger.warning("No valid features for %s/%s/%s", mk, country, fs_name)
                    continue

                X_train = train_df[features].fillna(0)
                y_train = train_df["GHI_kWh_m2"]
                X_test  = test_df[features].fillna(0)
                y_test  = test_df["GHI_kWh_m2"]

                done += 1
                logger.info(
                    "[%d/%d]  %s | %s | %s  (%d features)",
                    done, total, model_name, country.upper(), fs_name, len(features),
                )

                t0    = time.time()
                model = MODEL_REGISTRY[mk]()
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    elapsed = time.time() - t0

                    # Neural networks output fewer rows (lookback offset)
                    y_test_aligned = y_test.values[-len(y_pred):]

                    result = evaluate_model(
                        y_true=y_test_aligned,
                        y_pred=y_pred,
                        model_name=model_name,
                        country=country,
                        feature_set=fs_name,
                    )
                    result["TrainTime_s"] = round(elapsed, 2)
                    all_results.append(result)

                    key = (model_name, country, fs_name)
                    predictions[key] = {"y_test": y_test_aligned, "y_pred": y_pred}

                    # Feature importances for tree models
                    if hasattr(model, "feature_importances_") and fs_name == "IMPORTANT":
                        feature_importances[(model_name, country)] = {
                            "features":    features,
                            "importances": model.feature_importances_,
                        }

                    # Save model
                    if save_models:
                        save_path = Path(MODEL_SAVE_DIR) / f"{mk}_{country}_{fs_name}.pkl"
                        model.save(save_path)

                except Exception as exc:  # pragma: no cover
                    logger.error("Training failed for %s/%s/%s: %s", mk, country, fs_name, exc)

    # ------------------------------------------------------------------
    # 5. Results
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5: Results summary")
    logger.info("=" * 60)

    results_df = pd.DataFrame(all_results).sort_values("R2", ascending=False).reset_index(drop=True)
    save_results(results_df, f"{RESULTS_DIR}/results.csv")
    print_summary(results_df)

    # ------------------------------------------------------------------
    # 6. Figures
    # ------------------------------------------------------------------
    if generate_figures:
        logger.info("=" * 60)
        logger.info("STEP 6: Generating figures")
        logger.info("=" * 60)
        save_all_figures(results_df, predictions, feature_importances, lstm_history)

    return results_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train solar radiation forecasting models for West Africa"
    )
    p.add_argument(
        "--models", nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=None,
        help="Models to train (default: all). Options: rf xgboost lstm cnn_lstm",
    )
    p.add_argument(
        "--countries", nargs="+",
        choices=COUNTRIES,
        default=None,
        help="Countries to include (default: all)",
    )
    p.add_argument(
        "--feature-sets", nargs="+",
        choices=["BASE", "IMPORTANT", "FULL"],
        default=None,
        dest="feature_sets",
        help="Feature sets to evaluate (default: all)",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Force re-download of raw data from energydata.info",
    )
    p.add_argument(
        "--no-figures", action="store_true",
        help="Skip figure generation",
    )
    p.add_argument(
        "--no-save-models", action="store_true",
        help="Do not save fitted models to disk",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiment(
        countries=args.countries,
        model_keys=args.models,
        feature_set_names=args.feature_sets,
        use_cache=not args.no_cache,
        save_models=not args.no_save_models,
        generate_figures=not args.no_figures,
    )
