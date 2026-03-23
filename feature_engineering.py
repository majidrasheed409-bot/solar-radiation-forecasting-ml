"""
feature_engineering.py
-----------------------
Generates all 81 engineered features from 11 base meteorological variables,
as described in dissertation Chapter 2.2.

Feature categories
------------------
  Temporal   (12)  – Month, Year, Quarter, Season, DayOfWeek, IsWeekend,
                      6 cyclical sine/cosine encodings
  Lag        (24)  – 6 variables × 4 lag periods (1, 7, 14, 30 days)
  Rolling    (36)  – 3 variables × 3 windows × 4 statistics
  Interaction (9)  – DTR, ratios, clearness/diffuse indices, combined indices
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from config.settings import (
    BASE_FEATURES,
    IMPORTANT_FEATURES,
    RANDOM_STATE,
    TARGET,
)

logger = logging.getLogger(__name__)

# Variables used in lag / rolling generation
_LAG_VARS     = ["GHI_kWh_m2", "DNI_kWh_m2", "DHI_kWh_m2",
                 "Temperature_C", "Humidity_%", "Wind_Speed_m_s"]
_LAG_PERIODS  = [1, 7, 14, 30]
_ROLL_VARS    = ["GHI_kWh_m2", "DNI_kWh_m2", "Temperature_C"]
_ROLL_WINDOWS = [7, 14, 30]
_SOLAR_CONST  = 1367.0   # W/m²


# ---------------------------------------------------------------------------
# Core engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline to *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Daily solar data with a ``Date`` column and base meteorological
        variables (as returned by ``data_ingestion.load_all_countries``).

    Returns
    -------
    pd.DataFrame
        Input extended with 81 new features.  Rows with NaN are dropped
        (arises from lag/rolling operations on the first 30 days).
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 1. Temporal features (12)
    # ------------------------------------------------------------------
    df["Month"]        = df["Date"].dt.month
    df["Year_Feature"] = df["Date"].dt.year
    df["Quarter"]      = df["Date"].dt.quarter
    df["DayOfWeek"]    = df["Date"].dt.dayofweek
    df["IsWeekend"]    = (df["DayOfWeek"] >= 5).astype(int)

    # West-African dry/wet season proxy (1 = wet, 0 = dry)
    wet_months = {4, 5, 6, 7, 8, 9}
    df["Season"] = df["Month"].isin(wet_months).astype(int)

    doy = df["Date"].dt.dayofyear
    df["Month_sin"]      = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"]      = np.cos(2 * np.pi * df["Month"] / 12)
    df["DayOfYear_sin"]  = np.sin(2 * np.pi * doy / 365)
    df["DayOfYear_cos"]  = np.cos(2 * np.pi * doy / 365)
    df["DayOfWeek_sin"]  = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfWeek_cos"]  = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    # ------------------------------------------------------------------
    # 2. Lag features (24)
    # ------------------------------------------------------------------
    for var in _LAG_VARS:
        if var not in df.columns:
            continue
        for lag in _LAG_PERIODS:
            df[f"{var}_lag{lag}"] = df[var].shift(lag)

    # ------------------------------------------------------------------
    # 3. Rolling statistics (36)
    # ------------------------------------------------------------------
    for var in _ROLL_VARS:
        if var not in df.columns:
            continue
        for win in _ROLL_WINDOWS:
            roll = df[var].rolling(win)
            df[f"{var}_roll{win}_mean"] = roll.mean()
            df[f"{var}_roll{win}_std"]  = roll.std()
            df[f"{var}_roll{win}_min"]  = roll.min()
            df[f"{var}_roll{win}_max"]  = roll.max()

    # ------------------------------------------------------------------
    # 4. Interaction / domain features (9)
    # ------------------------------------------------------------------
    # Diurnal Temperature Range
    if "Temp_Max_C" in df.columns and "Temp_Min_C" in df.columns:
        df["DTR"] = df["Temp_Max_C"] - df["Temp_Min_C"]
    else:
        df["DTR"] = 0.0

    df["Temp_Range_7d"]  = (
        df["Temperature_C"].rolling(7).max()
        - df["Temperature_C"].rolling(7).min()
        if "Temperature_C" in df.columns else 0.0
    )
    df["Humidity_Temp"]  = (
        df["Humidity_%"] * df["Temperature_C"] / 100
        if all(c in df.columns for c in ("Humidity_%", "Temperature_C")) else 0.0
    )
    df["Wind_Temp"]      = (
        df["Wind_Speed_m_s"] * df["Temperature_C"]
        if all(c in df.columns for c in ("Wind_Speed_m_s", "Temperature_C")) else 0.0
    )
    df["Wind_Humidity"]  = (
        df["Wind_Speed_m_s"] * df["Humidity_%"] / 100
        if all(c in df.columns for c in ("Wind_Speed_m_s", "Humidity_%")) else 0.0
    )

    # DNI / DHI ratio (cloud cover indicator)
    if "DNI_kWh_m2" in df.columns and "DHI_kWh_m2" in df.columns:
        df["DNI_DHI_Ratio"] = np.where(
            df["DHI_kWh_m2"] > 0,
            df["DNI_kWh_m2"] / df["DHI_kWh_m2"],
            0.0,
        )
        df["GHI_DNI_Ratio"] = np.where(
            df["DNI_kWh_m2"] > 0,
            df["GHI_kWh_m2"] / df["DNI_kWh_m2"],
            0.0,
        )
    else:
        df["DNI_DHI_Ratio"] = 0.0
        df["GHI_DNI_Ratio"] = 0.0

    # Clearness index (atmospheric transmittance)
    extra_terrestrial = _SOLAR_CONST * (1 + 0.033 * np.cos(2 * np.pi * doy / 365))
    df["Clearness_Index"] = np.where(
        extra_terrestrial > 0,
        df["GHI_kWh_m2"] / extra_terrestrial,
        0.0,
    ).clip(0, 1)

    # Diffuse fraction
    df["Diffuse_Fraction"] = np.where(
        df["GHI_kWh_m2"] > 0,
        df["DHI_kWh_m2"] / df["GHI_kWh_m2"],
        0.0,
    ).clip(0, 1) if "DHI_kWh_m2" in df.columns else 0.0

    # ------------------------------------------------------------------
    # 5. Drop rows that contain NaN (from lag/rolling head)
    # ------------------------------------------------------------------
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.debug("Feature engineering: %d → %d rows (dropped %d)", before, len(df), before - len(df))

    return df


def get_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Return BASE / IMPORTANT / FULL feature lists for *df*.

    The FULL set is built dynamically from column names (excludes
    Date, Country, and the target).

    Parameters
    ----------
    df : pd.DataFrame
        Engineered DataFrame (output of ``engineer_features``).

    Returns
    -------
    dict with keys "BASE", "IMPORTANT", "FULL".
    """
    exclude = {"Date", "Country", TARGET}
    full_features = [c for c in df.columns if c not in exclude]

    base_available = [f for f in BASE_FEATURES if f in df.columns and f != TARGET]
    imp_available  = [f for f in IMPORTANT_FEATURES if f in df.columns]

    return {
        "BASE":      base_available,
        "IMPORTANT": imp_available,
        "FULL":      full_features,
    }


def select_features_by_mutual_info(
    df: pd.DataFrame,
    n_features: int = 20,
    random_state: int = RANDOM_STATE,
) -> list[str]:
    """
    Select top *n_features* by mutual information with GHI.

    Useful for re-running feature selection on new data.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered DataFrame.
    n_features : int
        How many features to return.

    Returns
    -------
    list[str]
    """
    exclude = {"Date", "Country", TARGET}
    candidates = [c for c in df.columns if c not in exclude]
    X = df[candidates].fillna(0)
    y = df[TARGET]

    mi = mutual_info_regression(X, y, random_state=random_state)
    mi_series = pd.Series(mi, index=candidates).sort_values(ascending=False)
    top = mi_series.head(n_features).index.tolist()
    logger.info("Top %d features by mutual information: %s", n_features, top)
    return top


# ---------------------------------------------------------------------------
# Train / test split (chronological – no data leakage)
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split *df* chronologically (no shuffling) to prevent leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered DataFrame sorted by Date.
    test_size : float
        Fraction reserved for testing.

    Returns
    -------
    (train_df, test_df)
    """
    cutoff = int(len(df) * (1 - test_size))
    train = df.iloc[:cutoff].reset_index(drop=True)
    test  = df.iloc[cutoff:].reset_index(drop=True)
    logger.info(
        "Train: %d rows  (%s – %s) | Test: %d rows  (%s – %s)",
        len(train), train["Date"].min().date(), train["Date"].max().date(),
        len(test),  test["Date"].min().date(),  test["Date"].max().date(),
    )
    return train, test
