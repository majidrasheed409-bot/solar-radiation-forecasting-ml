"""
Solar Forecasting System - Configuration
MSc Dissertation: ML for Daily Solar Radiation Forecasting in West Africa
Author: Majid Rasheed, University of Hull, 2025
"""

# =============================================================================
# DATA SOURCE
# =============================================================================
ENERGYDATA_API_BASE = "https://energydata.info/api/3/action/"

DATASET_IDS = {
    "nigeria": [
        "solar-radiation-measurement-data-for-bauchi-state-nigeria",
        "solar-radiation-measurement-data-for-kano-state-nigeria",
    ],
    "ghana": [
        "solar-radiation-measurement-data-for-navrongo-ghana",
        "solar-radiation-measurement-data-for-sunyani-ghana",
    ],
    "senegal": [
        "solar-radiation-measurement-data-for-ourossogui-senegal",
        "solar-radiation-measurement-data-for-tambacounda-senegal",
    ],
}

COLUMN_MAPPING = {
    "GHI": "GHI_kWh_m2",
    "Global Horizontal Irradiance (GHI)": "GHI_kWh_m2",
    "DNI": "DNI_kWh_m2",
    "Direct Normal Irradiance (DNI)": "DNI_kWh_m2",
    "DHI": "DHI_kWh_m2",
    "Diffuse Horizontal Irradiance (DHI)": "DHI_kWh_m2",
    "Temperature": "Temperature_C",
    "Ambient Temperature": "Temperature_C",
    "Relative Humidity": "Humidity_%",
    "RH": "Humidity_%",
    "Wind Speed": "Wind_Speed_m_s",
    "WS": "Wind_Speed_m_s",
    "Wind Direction": "Wind_Direction_deg",
    "WD": "Wind_Direction_deg",
    "Barometric Pressure": "Barometric_Pressure_hPa",
    "BP": "Barometric_Pressure_hPa",
    "Precipitation": "Precipitation_mm",
    "Rain": "Precipitation_mm",
}

# =============================================================================
# FEATURE SETS  (from dissertation Chapter 2 & Appendix)
# =============================================================================
TARGET = "GHI_kWh_m2"

BASE_FEATURES = [
    "GHI_kWh_m2", "DNI_kWh_m2", "DHI_kWh_m2",
    "Temperature_C", "Temp_Max_C", "Temp_Min_C",
    "Humidity_%", "Wind_Speed_m_s", "Wind_Direction_deg",
    "Barometric_Pressure_hPa", "Precipitation_mm",
]

# Top-20 selected via mutual information (dissertation §2.2)
IMPORTANT_FEATURES = [
    "DNI_kWh_m2", "Month", "DHI_kWh_m2",
    "GHI_kWh_m2_lag1", "DHI_kWh_m2_lag1", "Year_Feature",
    "DHI_kWh_m2_lag7", "Wind_Direction_deg", "DNI_kWh_m2_lag1",
    "GHI_kWh_m2_lag14", "Wind_Speed_m_s", "DNI_kWh_m2_lag30",
    "GHI_kWh_m2_lag7", "DNI_kWh_m2_lag7", "GHI_kWh_m2_lag30",
    "GHI_kWh_m2_roll7_mean", "Temperature_C", "Clearness_Index",
    "DNI_DHI_Ratio", "Humidity_%",
]

# =============================================================================
# MODEL HYPERPARAMETERS  (from dissertation §2.3 – best validated configs)
# =============================================================================
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 5,
    "random_state": 42,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

LSTM_PARAMS = {
    "units": 32,
    "dropout": 0.2,
    "epochs": 100,
    "batch_size": 32,
    "patience": 15,
    "lookback": 7,   # days
}

CNN_LSTM_PARAMS = {
    "filters": 32,
    "lstm_units": 32,
    "dropout": 0.2,
    "epochs": 100,
    "batch_size": 32,
    "patience": 15,
    "lookback": 7,
}

# =============================================================================
# EVALUATION THRESHOLDS
# =============================================================================
PERFORMANCE_THRESHOLDS = {
    "excellent": 0.95,
    "good":      0.90,
    "acceptable": 0.80,
}

COUNTRIES = list(DATASET_IDS.keys())
RANDOM_STATE = 42
TEST_SIZE = 0.20          # 80/20 chronological split
DATA_CACHE_DIR = "data_cache"
MODEL_SAVE_DIR = "saved_models"
RESULTS_DIR    = "results"
FIGURES_DIR    = "figures"
