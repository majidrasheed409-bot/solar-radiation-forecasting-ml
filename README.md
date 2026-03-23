# Solar Radiation Forecasting – West Africa
**MSc Dissertation | Majid Rasheed | University of Hull | December 2025**

Machine Learning for Daily Solar Radiation Forecasting in West Africa:  
*A Comparative Analysis of Tree-Based and Deep Learning Approaches*

---

## Project Structure

```
solar_forecasting/
├── config/
│   └── settings.py          # All hyperparameters, feature sets, paths
├── src/
│   ├── data_ingestion.py    # World Bank energydata.info API + caching
│   ├── feature_engineering.py  # 81-variable feature pipeline
│   ├── models.py            # RF, XGBoost, LSTM, CNN-LSTM classes
│   ├── evaluation.py        # R², RMSE, MAE, MAPE + reporting
│   └── visualisation.py     # Figures 1–9 from the dissertation
├── scripts/
│   ├── train.py             # Full training pipeline (CLI)
│   └── predict.py           # Inference on new data (CLI)
├── tests/
│   └── test_pipeline.py     # Pytest unit tests
├── data_cache/              # Auto-created: cached CSV downloads
├── saved_models/            # Auto-created: serialised model .pkl files
├── results/                 # Auto-created: results.csv
├── figures/                 # Auto-created: all 9 dissertation figures
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU users:** replace `tensorflow` with `tensorflow-gpu` in `requirements.txt`.

### 2. Train all models (downloads data automatically)
```bash
python scripts/train.py
```

This runs the full 24-experiment matrix (4 models × 3 countries × 2 feature sets),
saves results to `results/results.csv`, and generates all figures to `figures/`.

### 3. Train only tree-based models (fastest, best results)
```bash
python scripts/train.py --models rf xgboost --feature-sets IMPORTANT
```

### 4. Predict on new data
```bash
python scripts/predict.py \
    --model-path saved_models/xgboost_nigeria_IMPORTANT.pkl \
    --input data_cache/data_Nigeria_2021_2023.csv \
    --country nigeria \
    --output results/predictions.csv
```

### 5. Quick smoke test (validates a saved model)
```bash
python scripts/predict.py --smoke-test --country nigeria
```

### 6. Run tests
```bash
pytest tests/ -v
```

---

## CLI Reference

### `scripts/train.py`

| Argument | Default | Description |
|---|---|---|
| `--models` | all | `rf xgboost lstm cnn_lstm` |
| `--countries` | all | `nigeria ghana senegal` |
| `--feature-sets` | all | `BASE IMPORTANT FULL` |
| `--no-cache` | False | Force re-download from energydata.info |
| `--no-figures` | False | Skip figure generation |
| `--no-save-models` | False | Do not persist models to disk |

### `scripts/predict.py`

| Argument | Description |
|---|---|
| `--model-path` | Path to a `.pkl` saved model |
| `--input` | CSV of new daily meteorological data |
| `--country` | `nigeria` / `ghana` / `senegal` |
| `--feature-set` | `BASE` / `IMPORTANT` / `FULL` (default: `IMPORTANT`) |
| `--output` | Output CSV path (default: `results/predictions.csv`) |
| `--smoke-test` | Quick validation with cached data |

---

## Module API

```python
from src.data_ingestion      import load_all_countries
from src.feature_engineering import engineer_features, get_feature_sets, chronological_split
from src.models              import XGBoostModel, RandomForestModel
from src.evaluation          import evaluate_model, print_summary
from src.visualisation       import save_all_figures

# 1. Load data
solar_data = load_all_countries(use_cache=True)

# 2. Engineer features
df_eng = engineer_features(solar_data["nigeria"])

# 3. Split
train_df, test_df = chronological_split(df_eng, test_size=0.20)

# 4. Select features
fs     = get_feature_sets(df_eng)
feats  = fs["IMPORTANT"]
X_train, y_train = train_df[feats], train_df["GHI_kWh_m2"]
X_test,  y_test  = test_df[feats],  test_df["GHI_kWh_m2"]

# 5. Train & evaluate
model = XGBoostModel()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
result = evaluate_model(y_test, y_pred, "XGBoost", "nigeria", "IMPORTANT")
print(result)
# → {'R2': 0.9777, 'RMSE': 7.41, 'MAE': ..., 'MAPE': ..., 'Rating': 'Excellent'}

# 6. Save / load model
model.save("saved_models/xgboost_nigeria_IMPORTANT.pkl")
model2 = XGBoostModel.load("saved_models/xgboost_nigeria_IMPORTANT.pkl")
```

---

## Key Results (Dissertation Chapter 3)

| Model | Best R² | Config | RMSE (kWh/m²) |
|---|---|---|---|
| **XGBoost** | **0.9777** | Nigeria – IMPORTANT | **7.41** |
| Random Forest | 0.9728 | Nigeria – IMPORTANT | 8.19 |
| LSTM | 0.0523 | All configs | — |
| CNN-LSTM | 0.0525 | All configs | — |

Neural networks are included for completeness; their poor performance is attributable
to the limited training set size (~570 samples vs the ~17,000 minimum recommended
for the LSTM architecture employed — see §4.2).

---

## Feature Engineering Summary (81 features)

| Category | Count | Examples |
|---|---|---|
| Temporal | 12 | Month, Quarter, Season, DayOfYear_sin/cos |
| Lag | 24 | GHI_lag1, GHI_lag7, DNI_lag1, Temp_lag30 |
| Rolling | 36 | GHI_roll7_mean, GHI_roll14_std, DNI_roll30_max |
| Interaction | 9 | Clearness_Index, DNI_DHI_Ratio, Diffuse_Fraction, DTR |

The **IMPORTANT** set (top-20 by mutual information) consistently outperforms
both BASE (11) and FULL (81), reflecting the bias-variance optimum at
intermediate feature complexity.

---

## Data Source

World Bank energydata.info platform – "Solar Development in Sub-Saharan Africa"  
Stations: Bauchi & Kano (Nigeria) · Navrongo & Sunyani (Ghana) · Ourossogui & Tambacounda (Senegal)  
Period: September 2021 – November 2023 | Resolution: daily aggregated

---

## Deployment Recommendations (Dissertation §5.2)

1. **Primary model:** XGBoost + IMPORTANT feature set
2. **Ensemble partner:** Random Forest for robustness
3. **Retraining schedule:** Quarterly, aligned with seasonal transitions
4. **Hardware requirement:** Standard desktop CPU (<3 s training, <500 MB RAM)
5. **Neural networks:** Revisit when ≥17,000 training samples are available

---

*University of Hull · MSc AI and Data Science · 2025*
