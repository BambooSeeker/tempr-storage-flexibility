# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from shared_data import FORECAST_HORIZON, OUTPUT_DIR, build_dataset


MODEL_NAME = "SVR_PriceLags"

SELECTED_FEATURES = [
    "price_day_ahead_lag48",
    "price_day_ahead_lag96",
    "price_day_ahead_lag144",
    "price_day_ahead_lag336",
]


def select_target_window_features(
    X_flat: np.ndarray,
    all_feature_names: list[str],
    selected_features: list[str],
) -> np.ndarray:
    missing = [name for name in selected_features if name not in all_feature_names]
    if missing:
        raise ValueError(f"Selected features are missing from dataset: {missing}")

    selected_indices = [all_feature_names.index(name) for name in selected_features]
    Xw = X_flat.reshape(len(X_flat), FORECAST_HORIZON, len(all_feature_names))
    return Xw[:, :, selected_indices].reshape(len(X_flat), -1).astype(np.float32)


def expanded_feature_names(selected_features: list[str]) -> list[str]:
    names = []
    for step in range(FORECAST_HORIZON):
        names.extend([f"h{step + 1:02d}_{name}" for name in selected_features])
    return names


def make_model() -> MultiOutputRegressor:
    base = LinearSVR(
        C=0.03,
        epsilon=0.01,
        loss="squared_epsilon_insensitive",
        max_iter=8000,
        dual=False,
        tol=1e-3,
        random_state=42,
    )
    return MultiOutputRegressor(base, n_jobs=4)


def save_predictions(pred: np.ndarray, test_times: np.ndarray) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(pred, columns=[f"pred_t+{i + 1}" for i in range(FORECAST_HORIZON)])
    out.insert(0, "forecast_start_time", test_times)
    path = OUTPUT_DIR / f"pred_{MODEL_NAME}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_coefficients(model: MultiOutputRegressor, feature_names: list[str]) -> Path:
    coefs = [np.abs(estimator.coef_) for estimator in model.estimators_]
    importance = np.mean(np.vstack(coefs), axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_coef": importance})
    df = df.sort_values("mean_abs_coef", ascending=False)
    path = OUTPUT_DIR / f"feature_importance_{MODEL_NAME}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    data = build_dataset(save_outputs=True, verbose=True)

    X_train_full_raw = np.concatenate([data.X_flat_train, data.X_flat_val], axis=0)
    y_train_full = np.concatenate([data.y_train_scaled, data.y_val_scaled], axis=0)

    X_train_full = select_target_window_features(X_train_full_raw, data.feature_names, SELECTED_FEATURES)
    X_test = select_target_window_features(data.X_flat_test, data.feature_names, SELECTED_FEATURES)
    feature_names = expanded_feature_names(SELECTED_FEATURES)

    x_scaler = StandardScaler()
    X_train_full = x_scaler.fit_transform(X_train_full)
    X_test = x_scaler.transform(X_test)

    model = make_model()
    start = time.time()
    model.fit(X_train_full, y_train_full)
    train_seconds = time.time() - start

    start = time.time()
    pred_scaled = model.predict(X_test)
    pred_seconds = time.time() - start
    pred = data.scaler.inverse_transform(pred_scaled)

    if not np.all(np.isfinite(pred)):
        bad = int(np.sum(~np.isfinite(pred)))
        print(f"Warning: replacing {bad} non-finite predictions with 0.")
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

    mae = mean_absolute_error(data.y_test_raw, pred)
    rmse = mean_squared_error(data.y_test_raw, pred) ** 0.5
    baseline_mae = mean_absolute_error(data.y_test_raw, data.previous_day_baseline)
    skill_score = 1.0 - mae / baseline_mae if baseline_mae > 0 else 0.0

    pred_path = save_predictions(pred, data.test_times)
    importance_path = save_coefficients(model, feature_names)

    metrics = {
        "model": MODEL_NAME,
        "input_mode": "target_window_selected_price_lag_features",
        "selected_features": SELECTED_FEATURES,
        "mae": float(mae),
        "rmse": float(rmse),
        "baseline_previous_day_mae": float(baseline_mae),
        "skill_score_vs_previous_day": float(skill_score),
        "train_seconds": float(train_seconds),
        "pred_seconds": float(pred_seconds),
        "train_shape": list(X_train_full.shape),
        "test_shape": list(X_test.shape),
        "prediction_file": str(pred_path),
        "feature_importance_file": str(importance_path),
    }
    metrics_path = OUTPUT_DIR / f"metrics_{MODEL_NAME}.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{MODEL_NAME} done")
    print(f"  input: X[t:t+48] selected price-lag features, {len(SELECTED_FEATURES)} features per step")
    print(f"  selected features: {SELECTED_FEATURES}")
    print(f"  train shape: {X_train_full.shape}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  previous-day baseline MAE: {baseline_mae:.4f}")
    print(f"  skill score: {skill_score:.4f}")
    print(f"  train seconds: {train_seconds:.2f}")
    print(f"  pred seconds: {pred_seconds:.2f}")
    print(f"  saved: {pred_path}")


if __name__ == "__main__":
    main()
