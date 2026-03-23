"""
tests/test_pipeline.py
----------------------
Unit tests for the solar forecasting pipeline.
Run with:  pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import evaluate_model
from src.feature_engineering import (
    chronological_split,
    engineer_features,
    get_feature_sets,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """80 days of synthetic solar data (enough for 30-day rolling features)."""
    np.random.seed(42)
    n = 80
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Date":                   dates,
            "GHI_kWh_m2":            np.random.uniform(100, 300, n),
            "DNI_kWh_m2":            np.random.uniform(50,  250, n),
            "DHI_kWh_m2":            np.random.uniform(20,  100, n),
            "Temperature_C":         np.random.uniform(20,  40,  n),
            "Temp_Max_C":            np.random.uniform(35,  45,  n),
            "Temp_Min_C":            np.random.uniform(15,  25,  n),
            "Humidity_%":            np.random.uniform(30,  80,  n),
            "Wind_Speed_m_s":        np.random.uniform(1,   8,   n),
            "Wind_Direction_deg":    np.random.uniform(0,   360, n),
            "Barometric_Pressure_hPa": np.random.uniform(980, 1020, n),
            "Precipitation_mm":      np.random.uniform(0,   10,  n),
            "Country":               "Nigeria",
        }
    )


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    def test_output_has_more_columns(self, minimal_df):
        df_eng = engineer_features(minimal_df)
        assert len(df_eng.columns) > len(minimal_df.columns), \
            "Engineered df should have more columns than raw df"

    def test_no_nans_after_engineering(self, minimal_df):
        df_eng = engineer_features(minimal_df)
        assert df_eng.isnull().sum().sum() == 0, \
            "No NaN values should remain after engineer_features()"

    def test_row_count_reduced(self, minimal_df):
        df_eng = engineer_features(minimal_df)
        # First 30 rows dropped due to 30-day rolling features
        assert len(df_eng) < len(minimal_df), \
            "Rows should be fewer after dropping NaN from lag/rolling head"
        assert len(df_eng) > 0, "Some rows must survive"

    def test_required_temporal_features(self, minimal_df):
        df_eng = engineer_features(minimal_df)
        for col in ("Month", "Quarter", "Season", "Month_sin", "Month_cos",
                    "DayOfYear_sin", "DayOfYear_cos"):
            assert col in df_eng.columns, f"Missing temporal feature: {col}"

    def test_lag_features_created(self, minimal_df):
        df_eng = engineer_features(minimal_df)
        for lag in (1, 7, 14, 30):
            assert f"GHI_kWh_m2_lag{lag}" in df_eng.columns, \
                f"Missing lag feature: GHI_kWh_m2_lag{lag}"

    def test_rolling_features_created(self, minimal_df):
        df_eng = engineer_features(minimal_df)
        for win in (7, 14, 30):
            assert f"GHI_kWh_m2_roll{win}_mean" in df_eng.columns, \
                f"Missing rolling feature: GHI_kWh_m2_roll{win}_mean"

    def test_interaction_features_created(self, minimal_df):
        df_eng = engineer_features(minimal_df)
        for col in ("DTR", "Clearness_Index", "DNI_DHI_Ratio", "Diffuse_Fraction"):
            assert col in df_eng.columns, f"Missing interaction feature: {col}"

    def test_clearness_index_in_range(self, minimal_df):
        df_eng = engineer_features(minimal_df)
        assert df_eng["Clearness_Index"].between(0, 1).all(), \
            "Clearness_Index must be in [0, 1]"

    def test_diffuse_fraction_in_range(self, minimal_df):
        df_eng = engineer_features(minimal_df)
        assert df_eng["Diffuse_Fraction"].between(0, 1).all(), \
            "Diffuse_Fraction must be in [0, 1]"


# ---------------------------------------------------------------------------
# Feature Sets
# ---------------------------------------------------------------------------

class TestFeatureSets:
    def test_all_sets_present(self, minimal_df):
        df_eng = engineer_features(minimal_df)
        fs = get_feature_sets(df_eng)
        assert set(fs.keys()) == {"BASE", "IMPORTANT", "FULL"}

    def test_full_is_largest(self, minimal_df):
        df_eng = engineer_features(minimal_df)
        fs = get_feature_sets(df_eng)
        assert len(fs["FULL"]) >= len(fs["IMPORTANT"]) >= len(fs["BASE"]), \
            "FULL ≥ IMPORTANT ≥ BASE in feature count"

    def test_target_excluded_from_all_sets(self, minimal_df):
        df_eng = engineer_features(minimal_df)
        fs = get_feature_sets(df_eng)
        for name, feats in fs.items():
            assert "GHI_kWh_m2" not in feats, \
                f"Target GHI_kWh_m2 must not appear in feature set {name}"

    def test_all_features_present_in_df(self, minimal_df):
        df_eng = engineer_features(minimal_df)
        fs = get_feature_sets(df_eng)
        for name, feats in fs.items():
            missing = [f for f in feats if f not in df_eng.columns]
            assert not missing, \
                f"Feature set {name} references columns not in df: {missing}"


# ---------------------------------------------------------------------------
# Train/Test Split
# ---------------------------------------------------------------------------

class TestChronologicalSplit:
    def test_split_ratio(self, minimal_df):
        df_eng = engineer_features(minimal_df)
        train, test = chronological_split(df_eng, test_size=0.2)
        total = len(train) + len(test)
        assert abs(len(test) / total - 0.2) < 0.05, \
            "Test fraction should be approximately 20%"

    def test_no_temporal_leakage(self, minimal_df):
        df_eng = engineer_features(minimal_df)
        train, test = chronological_split(df_eng, test_size=0.2)
        assert train["Date"].max() < test["Date"].min(), \
            "All training dates must precede all test dates (no leakage)"

    def test_no_overlap(self, minimal_df):
        df_eng = engineer_features(minimal_df)
        train, test = chronological_split(df_eng, test_size=0.2)
        common = set(train["Date"]).intersection(set(test["Date"]))
        assert not common, "Train and test sets must not share dates"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_perfect_prediction(self):
        y = np.array([100.0, 200.0, 150.0, 180.0])
        result = evaluate_model(y, y, "Test", "Nigeria", "IMPORTANT")
        assert result["R2"] == pytest.approx(1.0, abs=1e-6)
        assert result["RMSE"] == pytest.approx(0.0, abs=1e-6)

    def test_metrics_keys_present(self):
        y_true = np.random.uniform(100, 300, 50)
        y_pred = y_true + np.random.normal(0, 5, 50)
        result = evaluate_model(y_true, y_pred, "XGBoost", "Ghana", "FULL")
        for key in ("Model", "Country", "FeatureSet", "R2", "RMSE", "MAE", "MAPE", "Rating"):
            assert key in result, f"Missing key in result dict: {key}"

    def test_rating_excellent(self):
        y = np.linspace(100, 300, 100)
        noise = np.random.RandomState(0).normal(0, 2, 100)
        result = evaluate_model(y, y + noise)
        assert result["R2"] > 0.95
        assert result["Rating"] == "Excellent"

    def test_rating_poor(self):
        y_true = np.linspace(100, 300, 100)
        y_pred = np.random.RandomState(1).uniform(100, 300, 100)
        result = evaluate_model(y_true, y_pred)
        assert result["Rating"] == "Poor"

    def test_mape_zero_guard(self):
        """MAPE should not blow up when y_true contains zeros."""
        y_true = np.array([0.0, 100.0, 200.0])
        y_pred = np.array([10.0, 110.0, 190.0])
        result = evaluate_model(y_true, y_pred)
        assert np.isfinite(result["MAPE"])


# ---------------------------------------------------------------------------
# Model smoke-tests (train on tiny data, check predict shape)
# ---------------------------------------------------------------------------

class TestModelInterfaces:
    """Verify that all four model classes expose the expected API."""

    @pytest.fixture
    def tiny_data(self, minimal_df):
        df_eng = engineer_features(minimal_df)
        fs     = get_feature_sets(df_eng)
        feats  = [f for f in fs["IMPORTANT"] if f in df_eng.columns]
        train, test = chronological_split(df_eng, 0.2)
        X_train = train[feats].fillna(0)
        y_train = train["GHI_kWh_m2"]
        X_test  = test[feats].fillna(0)
        y_test  = test["GHI_kWh_m2"]
        return X_train, y_train, X_test, y_test

    def test_random_forest(self, tiny_data):
        from src.models import RandomForestModel
        X_train, y_train, X_test, y_test = tiny_data
        m = RandomForestModel(n_estimators=10)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        assert len(preds) == len(X_test)
        assert np.isfinite(preds).all()

    def test_xgboost(self, tiny_data):
        from src.models import XGBoostModel
        X_train, y_train, X_test, y_test = tiny_data
        m = XGBoostModel(n_estimators=10)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        assert len(preds) == len(X_test)
        assert np.isfinite(preds).all()

    def test_model_not_fitted_raises(self):
        from src.models import RandomForestModel
        m = RandomForestModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            m.predict(np.zeros((5, 3)))
