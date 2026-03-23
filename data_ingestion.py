"""
data_ingestion.py
-----------------
Downloads and preprocesses solar radiation data from the World Bank
energydata.info platform (CKAN API).

Pipeline:
  1. download_from_energydata_api()  – fetch single dataset CSV
  2. process_station_data()          – standardise columns + daily aggregate
  3. download_country_data()         – combine 2 stations, cache locally
  4. load_all_countries()            – load Nigeria / Ghana / Senegal
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from config.settings import (
    COLUMN_MAPPING,
    COUNTRIES,
    DATA_CACHE_DIR,
    DATASET_IDS,
    ENERGYDATA_API_BASE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_cache_path(country: str) -> Path:
    Path(DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    return Path(DATA_CACHE_DIR) / f"data_{country.title()}_2021_2023.csv"


def _download_csv(url: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
    """Download a CSV from *url* with retry logic."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            for enc in ("utf-8", "latin-1", "iso-8859-1"):
                try:
                    df = pd.read_csv(io.StringIO(resp.content.decode(enc)))
                    logger.debug("Downloaded %d rows from %s", len(df), url[:60])
                    return df
                except UnicodeDecodeError:
                    continue
        except requests.RequestException as exc:
            if attempt < max_retries - 1:
                logger.warning("Retry %d/%d for %s – %s", attempt + 1, max_retries, url[:60], exc)
                time.sleep(2)
            else:
                logger.error("Download failed after %d attempts: %s", max_retries, exc)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_from_energydata_api(dataset_id: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
    """
    Fetch a dataset from the World Bank energydata.info CKAN API.

    Parameters
    ----------
    dataset_id : str
        The dataset slug as registered on energydata.info.
    max_retries : int
        Number of HTTP retry attempts.

    Returns
    -------
    pd.DataFrame | None
        Raw (unaggregated) records, or None on failure.
    """
    logger.info("Fetching dataset: %s", dataset_id)
    try:
        resp = requests.get(
            f"{ENERGYDATA_API_BASE}package_show",
            params={"id": dataset_id},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.error("API request failed: %s", exc)
        return None

    if not data.get("success"):
        logger.error("API returned error for %s: %s", dataset_id, data.get("error"))
        return None

    resources = data["result"].get("resources", [])
    csv_url = next(
        (r.get("url") for r in resources if r.get("format", "").upper() == "CSV"),
        None,
    )
    if not csv_url:
        logger.warning("No CSV resource found for %s", dataset_id)
        return None

    return _download_csv(csv_url, max_retries=max_retries)


def process_station_data(df: pd.DataFrame, country: str) -> Optional[pd.DataFrame]:
    """
    Standardise raw station data and aggregate to daily averages.

    Steps
    -----
    * Parse datetime column (first date/time column, or first column).
    * Rename columns according to COLUMN_MAPPING.
    * Compute per-day temperature min/max.
    * Average all numeric columns to daily resolution.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data as returned by ``download_from_energydata_api``.
    country : str
        Country label to attach (used for tracking provenance).

    Returns
    -------
    pd.DataFrame | None
    """
    if df is None or df.empty:
        return None

    df = df.copy()

    # Parse datetime
    date_cols = [c for c in df.columns if any(kw in c.lower() for kw in ("date", "time"))]
    dt_col = date_cols[0] if date_cols else df.columns[0]
    df["DateTime"] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=["DateTime"])
    df["Date"] = df["DateTime"].dt.date

    # Standardise column names
    for old, new in COLUMN_MAPPING.items():
        matching = [c for c in df.columns if old.lower() in c.lower()]
        for col in matching:
            if new not in df.columns:
                df[new] = pd.to_numeric(df[col], errors="coerce")

    # Daily min/max temperature
    if "Temperature_C" in df.columns:
        temp_stats = (
            df.groupby("Date")["Temperature_C"]
            .agg(Temp_Max_C="max", Temp_Min_C="min")
            .reset_index()
        )

    # Daily averages
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    daily = df.groupby("Date")[numeric_cols].mean().reset_index()

    if "Temperature_C" in df.columns:
        daily = daily.merge(temp_stats, on="Date", how="left")

    daily["Date"] = pd.to_datetime(daily["Date"])
    daily["Country"] = country.title()
    return daily


def download_country_data(country: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
    """
    Download, merge (2 stations), and cache daily solar data for *country*.

    Parameters
    ----------
    country : str
        One of ``"nigeria"``, ``"ghana"``, ``"senegal"``.
    use_cache : bool
        If True and a local CSV cache exists, skip the API download.

    Returns
    -------
    pd.DataFrame | None
    """
    cache_path = _get_cache_path(country)

    if use_cache and cache_path.exists():
        logger.info("Loading cached data: %s", cache_path)
        df = pd.read_csv(cache_path)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    logger.info("Downloading %s data from energydata.info …", country.upper())
    station_frames: list[pd.DataFrame] = []

    for dataset_id in DATASET_IDS.get(country, []):
        raw = download_from_energydata_api(dataset_id)
        if raw is not None:
            processed = process_station_data(raw, country)
            if processed is not None:
                station_frames.append(processed)

    if not station_frames:
        logger.error("Failed to download any station data for %s", country)
        if cache_path.exists():
            df = pd.read_csv(cache_path)
            df["Date"] = pd.to_datetime(df["Date"])
            return df
        return None

    combined = pd.concat(station_frames, ignore_index=True)
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    final = combined.groupby("Date")[numeric_cols].mean().reset_index()
    final["Country"] = country.title()
    final["Date"] = pd.to_datetime(final["Date"])

    final.to_csv(cache_path, index=False)
    logger.info("Saved %d records to %s", len(final), cache_path)
    return final


def load_all_countries(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """
    Load data for all three target countries.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are country names; values are daily DataFrames.
    """
    solar_data: dict[str, pd.DataFrame] = {}
    for country in COUNTRIES:
        df = download_country_data(country, use_cache=use_cache)
        if df is not None:
            solar_data[country] = df
            logger.info(
                "%s: %d days  (%s → %s)",
                country.upper(),
                len(df),
                df["Date"].min().date(),
                df["Date"].max().date(),
            )
        else:
            logger.warning("%s: FAILED to load data", country.upper())
    return solar_data
