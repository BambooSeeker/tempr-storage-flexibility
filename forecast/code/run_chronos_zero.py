# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import time
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import ChronosBoltPipeline, ChronosPipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

from shared_data import FORECAST_HORIZON, OUTPUT_DIR, TARGET_COL, TIME_COL, build_dataset


MODEL_NAME = "Chronos_Zero"
DAILY_START_TIME = dtime(0, 30)
CONTEXT_LEN = 2048
MODEL_ID = os.environ.get("CHRONOS_MODEL_ID", "amazon/chronos-bolt-small")


def _cols(prefix: str) -> list[str]:
    return [f"{prefix}_t+{i + 1}" for i in range(FORECAST_HORIZON)]


def _history_before(df: pd.DataFrame, forecast_start_time: pd.Timestamp, context_len: int) -> np.ndarray:
    times = df[TIME_COL].to_numpy()
    idx = int(np.searchsorted(times, np.datetime64(forecast_start_time)))
    start = max(0, idx - context_len)
    hist = df.iloc[start:idx][TARGET_COL].to_numpy(dtype=np.float32)
    if len(hist) < FORECAST_HORIZON:
        raise ValueError(f"Too little Chronos history before {forecast_start_time}.")
    return hist


def _load_pipeline(model_id: str, device: torch.device):
    kwargs = {"device_map": str(device)}
    if device.type == "cuda":
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.float32

    try:
        return ChronosBoltPipeline.from_pretrained(model_id, **kwargs), "ChronosBoltPipeline"
    except Exception as bolt_error:
        fallback_id = model_id.replace("chronos-bolt", "chronos-t5")
        try:
            return ChronosPipeline.from_pretrained(fallback_id, **kwargs), "ChronosPipeline"
        except Exception as chronos_error:
            raise RuntimeError(
                f"Failed to load Chronos model {model_id!r} and fallback {fallback_id!r}."
            ) from chronos_error


def _point_forecast(pipeline, history: np.ndarray) -> np.ndarray:
    context = torch.tensor(history, dtype=torch.float32)
    forecast = pipeline.predict(context, prediction_length=FORECAST_HORIZON)
    arr = forecast.detach().cpu().numpy()
    if arr.ndim == 3:
        # Chronos-Bolt returns (batch, quantiles, horizon); quantile index 4 is q=0.5.
        pred = arr[0, arr.shape[1] // 2, :]
    elif arr.ndim == 2:
        # Classic Chronos returns samples; use sample median.
        pred = np.median(arr[0], axis=0) if arr.shape[0] == 1 else np.median(arr, axis=0)
    else:
        raise ValueError(f"Unexpected Chronos forecast shape: {arr.shape}")
    return pred.astype(np.float32)


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

    load_start = time.time()
    pipeline, pipeline_class = _load_pipeline(MODEL_ID, device)
    load_seconds = time.time() - load_start

    preds = []
    start = time.time()
    for i, t in enumerate(daily_times, start=1):
        history = _history_before(data.df, pd.Timestamp(t), CONTEXT_LEN)
        preds.append(_point_forecast(pipeline, history))
        print(f"forecast {i:03d}/{len(daily_times)} {pd.Timestamp(t)}")
    pred_seconds = time.time() - start
    pred = np.asarray(preds, dtype=np.float32)

    mae = mean_absolute_error(y_daily, pred)
    rmse = mean_squared_error(y_daily, pred) ** 0.5
    baseline_mae = mean_absolute_error(y_daily, baseline_daily)
    skill = 1.0 - mae / baseline_mae if baseline_mae > 0 else 0.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(pred, columns=_cols("pred"))
    out.insert(0, "forecast_start_time", daily_times)
    pred_path = OUTPUT_DIR / f"daily_pred_{MODEL_NAME}.csv"
    out.to_csv(pred_path, index=False, encoding="utf-8-sig")

    metrics = {
        "model": MODEL_NAME,
        "eval_mode": "official_daily_00:30",
        "model_id": MODEL_ID,
        "pipeline_class": pipeline_class,
        "context_len": CONTEXT_LEN,
        "daily_samples": int(len(y_daily)),
        "mae": float(mae),
        "rmse": float(rmse),
        "baseline_previous_day_mae": float(baseline_mae),
        "skill_score_vs_previous_day": float(skill),
        "load_seconds": float(load_seconds),
        "pred_seconds": float(pred_seconds),
        "device": str(device),
        "prediction_file": str(pred_path),
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
