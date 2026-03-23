"""
models.py
---------
RandomForestModel, XGBoostModel, LSTMModel, CNNLSTMModel.
All config values are inlined to avoid import issues on Streamlit Cloud.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


class _BaseModel:
    """Shared save/load logic."""
    name: str = "BaseModel"

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("Model saved: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "_BaseModel":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        logger.info("Model loaded: %s", path)
        return obj


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class RandomForestModel(_BaseModel):
    name = "Random Forest"

    def __init__(self, **rf_kwargs) -> None:
        params = {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 5,
            "random_state": 42,
            "n_jobs": -1,
            **rf_kwargs,
        }
        self.model  = RandomForestRegressor(**params)
        self.scaler = StandardScaler()
        self.is_fit = False

    def fit(self, X_train, y_train) -> "RandomForestModel":
        X = np.array(X_train)
        X_sc = self.scaler.fit_transform(X)
        self.model.fit(X_sc, np.array(y_train))
        self.is_fit = True
        logger.info("%s fitted on %d samples", self.name, X.shape[0])
        return self

    def predict(self, X_test) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("Model not fitted yet – call .fit() first.")
        X_sc = self.scaler.transform(np.array(X_test))
        return self.model.predict(X_sc)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBoostModel(_BaseModel):
    name = "XGBoost"

    def __init__(self, **xgb_kwargs) -> None:
        from xgboost import XGBRegressor  # deferred import
        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            **xgb_kwargs,
        }
        self.model  = XGBRegressor(**params)
        self.scaler = StandardScaler()
        self.is_fit = False

    def fit(self, X_train, y_train) -> "XGBoostModel":
        X = np.array(X_train)
        X_sc = self.scaler.fit_transform(X)
        self.model.fit(X_sc, np.array(y_train))
        self.is_fit = True
        logger.info("%s fitted on %d samples", self.name, X.shape[0])
        return self

    def predict(self, X_test) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("Model not fitted yet – call .fit() first.")
        X_sc = self.scaler.transform(np.array(X_test))
        return self.model.predict(X_sc)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_


# ---------------------------------------------------------------------------
# LSTM  (deferred TensorFlow import)
# ---------------------------------------------------------------------------

class LSTMModel(_BaseModel):
    name = "LSTM"

    def __init__(self, lookback: int = 7, **kwargs) -> None:
        self.params = {
            "units": 32, "dropout": 0.2, "epochs": 100,
            "batch_size": 32, "patience": 15, "lookback": lookback,
            **kwargs,
        }
        self.scaler = RobustScaler()
        self._model = None
        self.is_fit = False

    def _build(self, n_features: int):
        from tensorflow.keras.layers import Dense, Dropout, LSTM
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
        lookback = self.params["lookback"]
        model = Sequential([
            LSTM(self.params["units"], activation="tanh",
                 input_shape=(lookback, n_features)),
            Dropout(self.params["dropout"]),
            Dense(1),
        ])
        model.compile(optimizer=Adam(), loss="mse")
        return model

    def _make_sequences(self, X: np.ndarray) -> np.ndarray:
        lb = self.params["lookback"]
        return np.array([X[i - lb:i] for i in range(lb, len(X))])

    def fit(self, X_train, y_train) -> "LSTMModel":
        from tensorflow.keras.callbacks import EarlyStopping
        X = self.scaler.fit_transform(np.array(X_train))
        y = np.array(y_train)
        X_seq = self._make_sequences(X)
        y_seq = y[self.params["lookback"]:]
        self._model = self._build(X.shape[1])
        es = EarlyStopping(monitor="val_loss", patience=self.params["patience"],
                           restore_best_weights=True)
        self._model.fit(X_seq, y_seq, epochs=self.params["epochs"],
                        batch_size=self.params["batch_size"],
                        validation_split=0.1, callbacks=[es], verbose=0)
        self.is_fit = True
        return self

    def predict(self, X_test) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("Model not fitted yet – call .fit() first.")
        X = self.scaler.transform(np.array(X_test))
        return self._model.predict(self._make_sequences(X), verbose=0).flatten()


# ---------------------------------------------------------------------------
# CNN-LSTM  (deferred TensorFlow import)
# ---------------------------------------------------------------------------

class CNNLSTMModel(_BaseModel):
    name = "CNN-LSTM"

    def __init__(self, lookback: int = 7, **kwargs) -> None:
        self.params = {
            "filters": 32, "lstm_units": 32, "dropout": 0.2,
            "epochs": 100, "batch_size": 32, "patience": 15,
            "lookback": lookback, **kwargs,
        }
        self.scaler = RobustScaler()
        self._model = None
        self.is_fit = False

    def _build(self, n_features: int):
        from tensorflow.keras.layers import Conv1D, Dense, Dropout, LSTM
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
        lookback = self.params["lookback"]
        model = Sequential([
            Conv1D(filters=self.params["filters"], kernel_size=3,
                   activation="relu", input_shape=(lookback, n_features),
                   padding="same"),
            LSTM(self.params["lstm_units"], activation="tanh"),
            Dropout(self.params["dropout"]),
            Dense(1),
        ])
        model.compile(optimizer=Adam(), loss="mse")
        return model

    def _make_sequences(self, X: np.ndarray) -> np.ndarray:
        lb = self.params["lookback"]
        return np.array([X[i - lb:i] for i in range(lb, len(X))])

    def fit(self, X_train, y_train) -> "CNNLSTMModel":
        from tensorflow.keras.callbacks import EarlyStopping
        X = self.scaler.fit_transform(np.array(X_train))
        y = np.array(y_train)
        X_seq = self._make_sequences(X)
        y_seq = y[self.params["lookback"]:]
        self._model = self._build(X.shape[1])
        es = EarlyStopping(monitor="val_loss", patience=self.params["patience"],
                           restore_best_weights=True)
        self._model.fit(X_seq, y_seq, epochs=self.params["epochs"],
                        batch_size=self.params["batch_size"],
                        validation_split=0.1, callbacks=[es], verbose=0)
        self.is_fit = True
        return self

    def predict(self, X_test) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("Model not fitted yet – call .fit() first.")
        X = self.scaler.transform(np.array(X_test))
        return self._model.predict(self._make_sequences(X), verbose=0).flatten()
