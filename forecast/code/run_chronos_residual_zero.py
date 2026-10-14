# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import time
from datetime import time as dtime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from run_chronos_zero import _cols, _load_pipeline, _point_forecast
from shared_data import FORECAST_HORIZON, OUTPUT_DIR, TARGET_COL, TIME_COL, build_dataset


MODEL_NAME = "Chronos_ResidualZero"
DAILY_START_TIME = dtime(0, 30)
CONTEXT_LEN = 2048
MODEL_ID = os.environ.get("CHRONOS_MODEL_ID", "amazon/chronos-bolt-small")


def _add_same_day_residual(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_price_lag48"] = out[TARGET_COL].shift(FORECAST_HORIZON)
    out["_price_residual48"] = out[TARGET_COL] - out["_price_lag48"]
    return out


def _history_residual_before(df: pd.DataFrame, forecast_start_time: pd.Timestamp, context_len: int) -> np.ndarray:
    times = df[TIME_COL].to_numpy()
    idx = int(np.searchsorted(times, np.datetime64(forecast_start_time)))
    start = max(FORECAST_HORIZON, idx - context_len)
    hist = df.iloc[start:idx]["_price_residual48"].to_numpy(dtype=np.float32)
    hist = hist[np.isfinite(hist)]
    if len(hist) < FORECAST_HORIZON:
        raise ValueError(f"Too little Chronos residual history before {forecast_start_time}.")
    return hist


def main() -> None:
    data = build_dataset(save_outputs=True, verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_df = pd.DataFrame(data.y_test_raw, columns=_cols("target"))
    y_df.insert(0, "forecast_start_time", pd.to_datetime(data.test_times))
    daily_mask = y_df["forecast_start_time"].dt.time == DAILY_START_TIME
    daily_times = y_df.loc[daily_mask, "forecast_start_time"].reset_index(drop=True)
    y_daily = y_df.loc[daily_mask, _cols("target")].to_numpy(dtype=np.float32)

    baseline_df = pd.DataFrame(data.previous_day_baseline, columns=_cols("baseline"))
    baseline_daily = baseline_df.loc[daily_mask, _cols("baseline")].to_numpy(dtype=np.float32)

    residual_df = _add_same_day_residual(data.df)

    load_start = time.time()
    pipeline, pipeline_class = _load_pipeline(MODEL_ID, device)
    load_seconds = time.time() - load_start

    residual_preds = []
    start = time.time()
    for i, t in enumerate(daily_times, start=1):
        history = _history_residual_before(residual_df, pd.Timestamp(t), CONTEXT_LEN)
        residual_preds.append(_point_forecast(pipeline, history))
        print(f"forecast {i:03d}/{len(daily_times)} {pd.Timestamp(t)}")
    pred_seconds = time.time() - start

    residual_pred = np.asarray(residual_preds, dtype=np.float32)
    pred = (baseline_daily + residual_pred).astype(np.float32)

    mae = mean_absolute_error(y_daily, pred)
    rmse = mean_squared_error(y_daily, pred) ** 0.5
    baseline_mae = mean_absolute_error(y_daily, baseline_daily)
    skill = 1.0 - mae / baseline_mae if baseline_mae > 0 else 0.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(pred, columns=_cols("pred"))
    out.insert(0, "forecast_start_time", daily_times)
    pred_path = OUTPUT_DIR / f"daily_pred_{MODEL_NAME}.csv"
    out.to_csv(pred_path, index=False, encoding="utf-8-sig")

    residual_out = pd.DataFrame(residual_pred, columns=[f"residual_pred_t+{i + 1}" for i in range(FORECAST_HORIZON)])
    residual_out.insert(0, "forecast_start_time", daily_times)
    residual_path = OUTPUT_DIR / f"daily_residual_pred_{MODEL_NAME}.csv"
    residual_out.to_csv(residual_path, index=False, encoding="utf-8-sig")

    metrics = {
        "model": MODEL_NAME,
        "eval_mode": "official_daily_00:30",
        "model_id": MODEL_ID,
        "pipeline_class": pipeline_class,
        "context_len": CONTEXT_LEN,
        "target_mode": "predict_price_day_ahead_minus_price_day_ahead_lag48",
        "final_prediction": "previous_day_baseline_plus_predicted_residual",
        "daily_samples": int(len(y_daily)),
        "mae": float(mae),
        "rmse": float(rmse),
        "baseline_previous_day_mae": float(baseline_mae),
        "skill_score_vs_previous_day": float(skill),
        "load_seconds": float(load_seconds),
        "pred_seconds": float(pred_seconds),
        "device": str(device),
        "prediction_file": str(pred_path),
        "residual_prediction_file": str(residual_path),
    }
    metrics_path = OUTPUT_DIR / f"metrics_daily_{MODEL_NAME}.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{MODEL_NAME} done")
    print(f"  model_id: {MODEL_ID}")
    print(f"  pipeline: {pipeline_class}")
    print(f"  device: {device}")
    print(f"  context_len: {CONTEXT_LEN}")
    print(f"  daily samples: {len(y_daily)}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  previous-day baseline MAE: {baseline_mae:.4f}")
    print(f"  skill score: {skill:.4f}")
    print(f"  load seconds: {load_seconds:.2f}")
    print(f"  pred seconds: {pred_seconds:.2f}")
    print(f"  saved: {pred_path}")


if __name__ == "__main__":
    main()
