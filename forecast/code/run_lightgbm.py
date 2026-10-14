# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

from shared_data import FORECAST_HORIZON, OUTPUT_DIR, build_dataset


MODEL_NAME = "LightGBM"

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


def make_model() -> MultiOutputRegressor:
    base = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    return MultiOutputRegressor(base, n_jobs=1)


def save_predictions(pred: np.ndarray, test_times: np.ndarray) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(pred, columns=[f"pred_t+{i+1}" for i in range(FORECAST_HORIZON)])
    out.insert(0, "forecast_start_time", test_times)
    path = OUTPUT_DIR / f"pred_{MODEL_NAME}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_feature_importance(model: MultiOutputRegressor, feature_names: list[str]) -> Path:
    n_features = model.estimators_[0].feature_importances_.shape[0]
    if n_features == len(feature_names):
        expanded_names = feature_names
    elif len(feature_names) > 0 and n_features % len(feature_names) == 0:
        n_steps = n_features // len(feature_names)
        expanded_names = [
            f"t+{step+1}_{name}"
            for step in range(n_steps)
            for name in feature_names
        ]
    else:
        expanded_names = [f"feature_{i}" for i in range(n_features)]

    importances = np.zeros(n_features, dtype=np.float64)
    for estimator in model.estimators_:
        importances += estimator.feature_importances_
    importances /= max(1, len(model.estimators_))

    df = pd.DataFrame({"feature": expanded_names, "importance": importances})
    df = df.sort_values("importance", ascending=False)
    path = OUTPUT_DIR / f"feature_importance_{MODEL_NAME}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    data = build_dataset(save_outputs=True, verbose=True)

    X_train_full = np.concatenate([data.X_flat_train, data.X_flat_val], axis=0)
    y_train_full = np.concatenate([data.y_train_scaled, data.y_val_scaled], axis=0)

    model = make_model()
    start = time.time()
    model.fit(X_train_full, y_train_full)
    train_seconds = time.time() - start

    start = time.time()
    pred_scaled = model.predict(data.X_flat_test)
    pred_seconds = time.time() - start
    pred = data.scaler.inverse_transform(pred_scaled)

    mae = mean_absolute_error(data.y_test_raw, pred)
    rmse = mean_squared_error(data.y_test_raw, pred) ** 0.5
    baseline_mae = mean_absolute_error(data.y_test_raw, data.previous_day_baseline)
    skill_score = 1.0 - mae / baseline_mae if baseline_mae > 0 else 0.0

    pred_path = save_predictions(pred, data.test_times)
    importance_path = save_feature_importance(model, data.feature_names)

    metrics = {
        "model": MODEL_NAME,
        "mae": float(mae),
        "rmse": float(rmse),
        "baseline_previous_day_mae": float(baseline_mae),
        "skill_score_vs_previous_day": float(skill_score),
        "train_seconds": float(train_seconds),
        "pred_seconds": float(pred_seconds),
        "train_shape": list(X_train_full.shape),
        "validation_shape_reserved": list(data.X_flat_val.shape),
        "test_shape": list(data.X_flat_test.shape),
        "prediction_file": str(pred_path),
        "feature_importance_file": str(importance_path),
    }
    metrics_path = OUTPUT_DIR / f"metrics_{MODEL_NAME}.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{MODEL_NAME} done")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  previous-day baseline MAE: {baseline_mae:.4f}")
    print(f"  skill score: {skill_score:.4f}")
    print(f"  train seconds: {train_seconds:.2f}")
    print(f"  pred seconds: {pred_seconds:.2f}")
    print(f"  saved: {pred_path}")


if __name__ == "__main__":
    main()
