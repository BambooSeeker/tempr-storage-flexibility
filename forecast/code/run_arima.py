# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
import warnings
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

from shared_data import FORECAST_HORIZON, OUTPUT_DIR, TARGET_COL, TIME_COL, build_dataset


MODEL_NAME = "ARIMA"
DAILY_START_TIME = dtime(0, 30)
CONTEXT_LEN = 336
ORDER = (1, 0, 1)
SEASONAL_ORDER = (1, 0, 0, 48)


def _cols(prefix: str) -> list[str]:
    return [f"{prefix}_t+{i + 1}" for i in range(FORECAST_HORIZON)]


def _history_before(df: pd.DataFrame, forecast_start_time: pd.Timestamp, context_len: int) -> np.ndarray:
    times = df[TIME_COL].to_numpy()
    idx = int(np.searchsorted(times, np.datetime64(forecast_start_time)))
    start = max(0, idx - context_len)
    history = df[TARGET_COL].to_numpy(dtype=np.float32)[start:idx]
    if len(history) < 8:
        raise ValueError(f"Too little ARIMA history before {forecast_start_time}.")
    return history


def _forecast_one(history: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = SARIMAX(
                history,
                order=ORDER,
                seasonal_order=SEASONAL_ORDER,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=50)
            pred = np.asarray(fit.forecast(steps=FORECAST_HORIZON), dtype=np.float32)
        except Exception:
            pred = np.full(FORECAST_HORIZON, float(np.mean(history)), dtype=np.float32)
    if not np.all(np.isfinite(pred)):
        pred = np.full(FORECAST_HORIZON, float(np.mean(history)), dtype=np.float32)
    return pred


def main() -> None:
    data = build_dataset(save_outputs=True, verbose=True)
    y_df = pd.DataFrame(data.y_test_raw, columns=_cols("target"))
    y_df.insert(0, "forecast_start_time", pd.to_datetime(data.test_times))
    daily_mask = y_df["forecast_start_time"].dt.time == DAILY_START_TIME
    daily_times = y_df.loc[daily_mask, "forecast_start_time"].reset_index(drop=True)
    y_daily = y_df.loc[daily_mask, _cols("target")].to_numpy(dtype=np.float32)

    baseline_df = pd.DataFrame(data.previous_day_baseline, columns=_cols("baseline"))
    baseline_daily = baseline_df.loc[daily_mask, _cols("baseline")].to_numpy(dtype=np.float32)

    preds = []
    start = time.time()
    for t in daily_times:
        history = _history_before(data.df, pd.Timestamp(t), CONTEXT_LEN)
        preds.append(_forecast_one(history))
    run_seconds = time.time() - start
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
        "order": ORDER,
        "seasonal_order": SEASONAL_ORDER,
        "context_len": CONTEXT_LEN,
        "daily_samples": int(len(y_daily)),
        "mae": float(mae),
        "rmse": float(rmse),
        "baseline_previous_day_mae": float(baseline_mae),
        "skill_score_vs_previous_day": float(skill),
        "run_seconds": float(run_seconds),
        "prediction_file": str(pred_path),
    }
    metrics_path = OUTPUT_DIR / f"metrics_daily_{MODEL_NAME}.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{MODEL_NAME} daily done")
    print(f"  order: {ORDER}")
    print(f"  seasonal_order: {SEASONAL_ORDER}")
    print(f"  context_len: {CONTEXT_LEN}")
    print(f"  daily samples: {len(y_daily)}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  previous-day baseline MAE: {baseline_mae:.4f}")
    print(f"  skill score: {skill:.4f}")
    print(f"  run seconds: {run_seconds:.2f}")
    print(f"  saved: {pred_path}")


if __name__ == "__main__":
    main()
