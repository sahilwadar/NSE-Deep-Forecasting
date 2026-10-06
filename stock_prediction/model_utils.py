"""Leakage-safe data preparation, LSTM training, evaluation, and persistence."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
MODEL_CACHE_VERSION = "v1"
ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class TrainingSettings:
    """Settings that affect how the two models are trained."""

    sequence_length: int = 60
    epochs: int = 30
    batch_size: int = 32
    units: int = 64
    patience: int = 5


@dataclass
class TrainingResult:
    """Predictions, metrics, and artifacts needed by the dashboard."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    forecast: dict[str, float]
    histories: dict[str, list[float]]
    winner: str
    train_rows: int
    test_rows: int
    cache_status: str


class ModelTrainingError(Exception):
    """An actionable error raised when the training dataset is too small."""


def _make_sequences(
    scaled_features: np.ndarray,
    scaled_target: np.ndarray,
    sequence_length: int,
    train_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create windows whose labels are strictly after each input window."""
    train_inputs: list[np.ndarray] = []
    train_targets: list[float] = []
    test_inputs: list[np.ndarray] = []
    test_targets: list[float] = []
    test_target_indexes: list[int] = []

    for target_index in range(sequence_length, len(scaled_features)):
        window = scaled_features[target_index - sequence_length : target_index]
        target_value = scaled_target[target_index, 0]

        if target_index < train_end:
            train_inputs.append(window)
            train_targets.append(target_value)
        else:
            # A test window can include earlier training-period observations:
            # those prices would already be known when making that prediction.
            test_inputs.append(window)
            test_targets.append(target_value)
            test_target_indexes.append(target_index)

    return (
        np.asarray(train_inputs, dtype=np.float32),
        np.asarray(train_targets, dtype=np.float32),
        np.asarray(test_inputs, dtype=np.float32),
        np.asarray(test_targets, dtype=np.float32),
        np.asarray(test_target_indexes, dtype=np.int64),
    )


def prepare_time_series(
    data: pd.DataFrame, sequence_length: int
) -> tuple[
    MinMaxScaler,
    MinMaxScaler,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    """Split chronologically, fit scalers on training rows, and build windows."""
    if sequence_length < 2:
        raise ModelTrainingError("Sequence length must be at least 2 trading days.")

    clean_data = data[FEATURE_COLUMNS].dropna().copy()
    row_count = len(clean_data)
    train_end = int(row_count * 0.8)
    minimum_train_examples = 8
    minimum_test_examples = 5

    if train_end - sequence_length < minimum_train_examples:
        raise ModelTrainingError(
            f"Not enough training history for a {sequence_length}-day sequence. "
            "Choose a shorter sequence length or a wider date range."
        )
    if row_count - train_end < minimum_test_examples:
        raise ModelTrainingError(
            "Not enough observations in the 20% test period. Choose a wider "
            "date range."
        )

    # Crucially, both scalers see only rows in the first 80% of the timeline.
    # Applying these fitted transformations to later rows does not refit them.
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    train_features = clean_data.iloc[:train_end][FEATURE_COLUMNS]
    feature_scaler.fit(train_features)
    target_scaler.fit(clean_data.iloc[:train_end][["Close"]])

    scaled_features = feature_scaler.transform(clean_data[FEATURE_COLUMNS])
    scaled_target = target_scaler.transform(clean_data[["Close"]])
    x_train, y_train, x_test, y_test, test_indexes = _make_sequences(
        scaled_features, scaled_target, sequence_length, train_end
    )

    if len(x_train) < minimum_train_examples or len(x_test) < minimum_test_examples:
        raise ModelTrainingError(
            "The selected history does not contain enough training and test "
            "sequences. Increase the date range or reduce sequence length."
        )

    return (
        feature_scaler,
        target_scaler,
        x_train,
        y_train,
        x_test,
        y_test,
        test_indexes,
        train_end,
    )


def _build_lstm(input_shape: tuple[int, int], units: int) -> tf.keras.Model:
    """Build the standard two-layer LSTM model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(units, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(units),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(max(units // 2, 8), activation="relu"),
            tf.keras.layers.Dense(1),
        ],
        name="LSTM",
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def _build_bilstm(input_shape: tuple[int, int], units: int) -> tf.keras.Model:
    """Build the two-layer bidirectional LSTM model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units, return_sequences=True)
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(max(units // 2, 8), activation="relu"),
            tf.keras.layers.Dense(1),
        ],
        name="BiLSTM",
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def _dataset_fingerprint(
    ticker: str, data: pd.DataFrame, settings: TrainingSettings
) -> str:
    """Create a stable key so a model is reused only for matching inputs."""
    digest = hashlib.sha256()
    digest.update(MODEL_CACHE_VERSION.encode("utf-8"))
    digest.update(ticker.encode("utf-8"))
    digest.update(json.dumps(asdict(settings), sort_keys=True).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(data, index=True).values.tobytes())
    return digest.hexdigest()[:20]


def _model_directory(
    model_root: Path, ticker: str, fingerprint: str
) -> Path:
    safe_ticker = re.sub(r"[^A-Za-z0-9_-]", "_", ticker)
    return model_root / f"{safe_ticker}_{fingerprint}"


def _load_saved_bundle(
    directory: Path,
) -> tuple[dict[str, tf.keras.Model], MinMaxScaler, MinMaxScaler, dict[str, list[float]]] | None:
    """Load a bundle only when its metadata and every expected file exist."""
    metadata_path = directory / "metadata.json"
    expected_files = [
        directory / "lstm.keras",
        directory / "bilstm.keras",
        directory / "feature_scaler.joblib",
        directory / "target_scaler.joblib",
        directory / "history.json",
    ]
    if not metadata_path.is_file() or not all(path.is_file() for path in expected_files):
        return None

    try:
        import joblib

        models = {
            "LSTM": tf.keras.models.load_model(directory / "lstm.keras"),
            "BiLSTM": tf.keras.models.load_model(directory / "bilstm.keras"),
        }
        feature_scaler = joblib.load(directory / "feature_scaler.joblib")
        target_scaler = joblib.load(directory / "target_scaler.joblib")
        histories = json.loads((directory / "history.json").read_text(encoding="utf-8"))
        return models, feature_scaler, target_scaler, histories
    except Exception:
        # An incomplete or incompatible local bundle is safe to rebuild.
        return None


def _save_bundle(
    directory: Path,
    models: dict[str, tf.keras.Model],
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    histories: dict[str, list[float]],
    settings: TrainingSettings,
    ticker: str,
) -> None:
    """Persist models, scalers, and loss curves for later runs."""
    import joblib

    directory.mkdir(parents=True, exist_ok=True)
    models["LSTM"].save(directory / "lstm.keras")
    models["BiLSTM"].save(directory / "bilstm.keras")
    joblib.dump(feature_scaler, directory / "feature_scaler.joblib")
    joblib.dump(target_scaler, directory / "target_scaler.joblib")
    (directory / "history.json").write_text(
        json.dumps(histories), encoding="utf-8"
    )
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "ticker": ticker,
                "settings": asdict(settings),
                "cache_version": MODEL_CACHE_VERSION,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _make_result(
    data: pd.DataFrame,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_indexes: np.ndarray,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    models: dict[str, tf.keras.Model],
    histories: dict[str, list[float]],
    train_end: int,
    cache_status: str,
) -> TrainingResult:
    """Generate aligned test predictions and score them in original price units."""
    actual_prices = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    prediction_data: dict[str, object] = {
        "Date": data.index[test_indexes],
        "Actual": actual_prices,
    }
    metric_rows: list[dict[str, float | str]] = []
    forecast: dict[str, float] = {}
    latest_window = data[FEATURE_COLUMNS].iloc[-x_test.shape[1] :]
    scaled_window = feature_scaler.transform(latest_window).reshape(
        1, x_test.shape[1], len(FEATURE_COLUMNS)
    )
    # One forward pass provides both the held-out predictions and final estimate.
    combined_inputs = np.concatenate(
        [x_test, scaled_window.astype(np.float32)], axis=0
    )

    for model_name, model in models.items():
        combined_predictions = model.predict(combined_inputs, verbose=0)
        scaled_predictions = combined_predictions[:-1]
        predictions = target_scaler.inverse_transform(scaled_predictions).ravel()
        prediction_data[model_name] = predictions

        nonzero_actual = np.where(actual_prices == 0, np.nan, actual_prices)
        mape = float(np.nanmean(np.abs((actual_prices - predictions) / nonzero_actual)) * 100)
        metric_rows.append(
            {
                "Model": model_name,
                "MAE": float(mean_absolute_error(actual_prices, predictions)),
                "RMSE": float(np.sqrt(mean_squared_error(actual_prices, predictions))),
                "MAPE (%)": mape,
            }
        )

        forecast[model_name] = float(
            target_scaler.inverse_transform(
                combined_predictions[-1:].reshape(-1, 1)
            )[0, 0]
        )

    metrics = pd.DataFrame(metric_rows).sort_values("RMSE").reset_index(drop=True)
    predictions_frame = pd.DataFrame(prediction_data)
    predictions_frame["Date"] = pd.to_datetime(predictions_frame["Date"])

    return TrainingResult(
        predictions=predictions_frame,
        metrics=metrics,
        forecast=forecast,
        histories=histories,
        winner=str(metrics.iloc[0]["Model"]),
        train_rows=train_end,
        test_rows=len(test_indexes),
        cache_status=cache_status,
    )


def train_or_load_models(
    data: pd.DataFrame,
    ticker: str,
    settings: TrainingSettings,
    model_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> TrainingResult:
    """Train both models or load a matching saved bundle, then evaluate it."""
    # Work from one clean frame so sequence indexes and chart dates remain aligned.
    clean_data = data[FEATURE_COLUMNS].dropna().copy()
    (
        feature_scaler,
        target_scaler,
        x_train,
        y_train,
        x_test,
        y_test,
        test_indexes,
        train_end,
    ) = prepare_time_series(clean_data, settings.sequence_length)

    fingerprint = _dataset_fingerprint(ticker, clean_data, settings)
    bundle_directory = _model_directory(model_root, ticker, fingerprint)
    saved_bundle = _load_saved_bundle(bundle_directory)

    if saved_bundle is None:
        try:
            # These settings improve repeatability across runs where possible.
            tf.keras.utils.set_random_seed(42)
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                pass

            input_shape = (settings.sequence_length, len(FEATURE_COLUMNS))
            models: dict[str, tf.keras.Model] = {
                "LSTM": _build_lstm(input_shape, settings.units),
                "BiLSTM": _build_bilstm(input_shape, settings.units),
            }
            histories: dict[str, list[float]] = {}
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=settings.patience,
                    restore_best_weights=True,
                )
            ]

            for model_number, (model_name, model) in enumerate(models.items(), start=1):
                callback = None
                if progress_callback is not None:
                    callback = _EpochProgressCallback(
                        model_name=model_name,
                        model_number=model_number,
                        total_epochs=settings.epochs,
                        on_epoch=progress_callback,
                    )

                history = model.fit(
                    x_train,
                    y_train,
                    validation_split=0.1,
                    epochs=settings.epochs,
                    batch_size=settings.batch_size,
                    shuffle=False,
                    callbacks=callbacks + ([callback] if callback is not None else []),
                    verbose=0,
                )
                histories[model_name] = [
                    float(value) for value in history.history.get("loss", [])
                ]

            _save_bundle(
                bundle_directory,
                models,
                feature_scaler,
                target_scaler,
                histories,
                settings,
                ticker,
            )
            cache_status = "Newly trained and saved"
        except ModelTrainingError:
            raise
        except Exception as exc:
            raise ModelTrainingError(
                "Model training did not finish successfully. Try fewer epochs, "
                "a wider date range, or a smaller sequence length."
            ) from exc
    else:
        models, feature_scaler, target_scaler, histories = saved_bundle
        cache_status = "Loaded from saved model bundle"
        if progress_callback is not None:
            progress_callback("Saved models loaded", 1, 1)

    return _make_result(
        data=clean_data,
        x_test=x_test,
        y_test=y_test,
        test_indexes=test_indexes,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        models=models,
        histories=histories,
        train_end=train_end,
        cache_status=cache_status,
    )


class _EpochProgressCallback(tf.keras.callbacks.Callback):
    """Bridge Keras epoch events to a small Streamlit progress callback."""

    def __init__(
        self,
        model_name: str,
        model_number: int,
        total_epochs: int,
        on_epoch: ProgressCallback,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.model_number = model_number
        self.total_epochs = total_epochs
        self.on_epoch = on_epoch

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        self.on_epoch(self.model_name, epoch + 1, self.total_epochs)


def set_global_seed(seed: int = 42) -> None:
    """Set Python and NumPy seeds before reproducible, non-TensorFlow steps."""
    import random

    random.seed(seed)
    np.random.seed(seed)
