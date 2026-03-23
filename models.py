"""
models.py
---------
Encapsulates training and prediction for all four model families:
  - RandomForestModel
  - XGBoostModel
  - LSTMModel
  - CNNLSTMModel

Each class exposes a consistent interface:
  .fit(X_train, y_train)
  .predict(X_test) → np.ndarray
  .save(path) / .load(path)

Tree-based models use StandardScaler; neural networks use RobustScaler.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: scaler-aware base class
# ---------------------------------------------------------------------------

class _BaseModel:
    """Shared save/load and scaler logic."""

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
    """
    Random Forest regressor with StandardScaler preprocessing.

    Parameters
    ----------
    **rf_kwargs
        Forwarded to ``sklearn.ensemble.RandomForestRegressor``.
    """

    name = "Random Forest"

    def __init__(self, **rf_kwargs) -> None:
        from config.settings import RF_PARAMS
        params = {**RF_PARAMS, **rf_kwargs}
        self.model   = RandomForestRegressor(**params)
        self.scaler  = StandardScaler()
        self.is_fit  = False

    def fit(self, X_train: pd.DataFrame | np.ndarray, y_train: pd.Series | np.ndarray) -> "RandomForestModel":
        X = np.array(X_train)
        X_sc = self.scaler.fit_transform(X)
        self.model.fit(X_sc, np.array(y_train))
        self.is_fit = True
        logger.info("%s fitted on %d samples, %d features", self.name, X.shape[0], X.shape[1])
        return self

    def predict(self, X_test: pd.DataFrame | np.ndarray) -> np.ndarray:
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
    """
    XGBoost regressor with StandardScaler preprocessing.

    Parameters
    ----------
    **xgb_kwargs
        Forwarded to ``xgboost.XGBRegressor``.
    """

    name = "XGBoost"

    def __init__(self, **xgb_kwargs) -> None:
        from config.settings import XGB_PARAMS
        params = {**XGB_PARAMS, **xgb_kwargs}
        self.model   = XGBRegressor(**params)
        self.scaler  = StandardScaler()
        self.is_fit  = False

    def fit(self, X_train: pd.DataFrame | np.ndarray, y_train: pd.Series | np.ndarray) -> "XGBoostModel":
        X = np.array(X_train)
        X_sc = self.scaler.fit_transform(X)
        self.model.fit(X_sc, np.array(y_train))
        self.is_fit = True
        logger.info("%s fitted on %d samples, %d features", self.name, X.shape[0], X.shape[1])
        return self

    def predict(self, X_test: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("Model not fitted yet – call .fit() first.")
        X_sc = self.scaler.transform(np.array(X_test))
        return self.model.predict(X_sc)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class LSTMModel(_BaseModel):
    """
    Single-layer LSTM with RobustScaler preprocessing and configurable lookback.

    Architecture  (dissertation §2.3)
    --------
    Input  → LSTM(32, tanh) → Dropout(0.2) → Dense(1)
    Training: Adam, MSE loss, EarlyStopping(patience=15)

    Parameters
    ----------
    lookback : int
        Number of previous time-steps fed as a sequence.
    **lstm_kwargs
        Override any key from ``config.settings.LSTM_PARAMS``.
    """

    name = "LSTM"

    def __init__(self, lookback: int = 7, **lstm_kwargs) -> None:
        from config.settings import LSTM_PARAMS
        self.params  = {**LSTM_PARAMS, **lstm_kwargs, "lookback": lookback}
        self.scaler  = RobustScaler()
        self._model  = None
        self.is_fit  = False

    # ---- Keras import is deferred to avoid import overhead if not used ------
    def _build(self, n_features: int):
        from tensorflow.keras.callbacks import EarlyStopping        # noqa
        from tensorflow.keras.layers import Dense, Dropout, LSTM    # noqa
        from tensorflow.keras.models import Sequential               # noqa
        from tensorflow.keras.optimizers import Adam                 # noqa

        lookback = self.params["lookback"]
        model = Sequential([
            LSTM(self.params["units"], activation="tanh",
                 input_shape=(lookback, n_features)),
            Dropout(self.params["dropout"]),
            Dense(1),
        ], name="LSTM_solar")
        model.compile(optimizer=Adam(), loss="mse")
        return model

    def _make_sequences(self, X: np.ndarray) -> np.ndarray:
        lookback = self.params["lookback"]
        seqs = np.array([X[i - lookback:i] for i in range(lookback, len(X))])
        return seqs

    def fit(self, X_train: pd.DataFrame | np.ndarray, y_train: pd.Series | np.ndarray) -> "LSTMModel":
        from tensorflow.keras.callbacks import EarlyStopping  # noqa

        X = self.scaler.fit_transform(np.array(X_train))
        y = np.array(y_train)

        lookback = self.params["lookback"]
        X_seq = self._make_sequences(X)
        y_seq = y[lookback:]

        self._model = self._build(X.shape[1])
        es = EarlyStopping(monitor="val_loss", patience=self.params["patience"],
                           restore_best_weights=True)
        self._model.fit(
            X_seq, y_seq,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            validation_split=0.1,
            callbacks=[es],
            verbose=0,
        )
        self.is_fit = True
        logger.info("%s fitted on %d sequences (lookback=%d)", self.name, len(X_seq), lookback)
        return self

    def predict(self, X_test: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fit or self._model is None:
            raise RuntimeError("Model not fitted yet – call .fit() first.")
        X = self.scaler.transform(np.array(X_test))
        X_seq = self._make_sequences(X)
        return self._model.predict(X_seq, verbose=0).flatten()


# ---------------------------------------------------------------------------
# CNN-LSTM
# ---------------------------------------------------------------------------

class CNNLSTMModel(_BaseModel):
    """
    Hybrid CNN-LSTM with RobustScaler preprocessing.

    Architecture  (dissertation §2.3)
    --------
    Input → Conv1D(32, kernel=3) → LSTM(32) → Dropout(0.2) → Dense(1)

    Parameters
    ----------
    lookback : int
        Sequence length fed to the model.
    **cnn_lstm_kwargs
        Override any key from ``config.settings.CNN_LSTM_PARAMS``.
    """

    name = "CNN-LSTM"

    def __init__(self, lookback: int = 7, **cnn_lstm_kwargs) -> None:
        from config.settings import CNN_LSTM_PARAMS
        self.params  = {**CNN_LSTM_PARAMS, **cnn_lstm_kwargs, "lookback": lookback}
        self.scaler  = RobustScaler()
        self._model  = None
        self.is_fit  = False

    def _build(self, n_features: int):
        from tensorflow.keras.layers import (  # noqa
            Conv1D, Dense, Dropout, LSTM, MaxPooling1D,
        )
        from tensorflow.keras.models import Sequential              # noqa
        from tensorflow.keras.optimizers import Adam                # noqa

        lookback = self.params["lookback"]
        model = Sequential([
            Conv1D(filters=self.params["filters"], kernel_size=3, activation="relu",
                   input_shape=(lookback, n_features), padding="same"),
            LSTM(self.params["lstm_units"], activation="tanh"),
            Dropout(self.params["dropout"]),
            Dense(1),
        ], name="CNN_LSTM_solar")
        model.compile(optimizer=Adam(), loss="mse")
        return model

    def _make_sequences(self, X: np.ndarray) -> np.ndarray:
        lookback = self.params["lookback"]
        return np.array([X[i - lookback:i] for i in range(lookback, len(X))])

    def fit(self, X_train: pd.DataFrame | np.ndarray, y_train: pd.Series | np.ndarray) -> "CNNLSTMModel":
        from tensorflow.keras.callbacks import EarlyStopping  # noqa

        X = self.scaler.fit_transform(np.array(X_train))
        y = np.array(y_train)

        lookback = self.params["lookback"]
        X_seq = self._make_sequences(X)
        y_seq = y[lookback:]

        self._model = self._build(X.shape[1])
        es = EarlyStopping(monitor="val_loss", patience=self.params["patience"],
                           restore_best_weights=True)
        self._model.fit(
            X_seq, y_seq,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            validation_split=0.1,
            callbacks=[es],
            verbose=0,
        )
        self.is_fit = True
        logger.info("%s fitted on %d sequences (lookback=%d)", self.name, len(X_seq), lookback)
        return self

    def predict(self, X_test: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fit or self._model is None:
            raise RuntimeError("Model not fitted yet – call .fit() first.")
        X = self.scaler.transform(np.array(X_test))
        X_seq = self._make_sequences(X)
        return self._model.predict(X_seq, verbose=0).flatten()
