# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import json
import logging
import os
import time
import warnings
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

from shared_data import FORECAST_HORIZON, OUTPUT_DIR, TARGET_COL, TIME_COL, build_dataset


MODEL_NAME = "Prophet"
DAILY_START_TIME = dtime(0, 30)
CONTEXT_LEN = 336


def _cols(prefix: str) -> list[str]:
    return [f"{prefix}_t+{i + 1}" for i in range(FORECAST_HORIZON)]


def _history_before(df: pd.DataFrame, forecast_start_time: pd.Timestamp, context_len: int) -> pd.DataFrame:
    times = df[TIME_COL].to_numpy()
    idx = int(np.searchsorted(times, np.datetime64(forecast_start_time)))
    start = max(0, idx - context_len)
    hist = df.iloc[start:idx][[TIME_COL, TARGET_COL]].copy()
    if len(hist) < 48:
        raise ValueError(f"Too little Prophet history before {forecast_start_time}.")
    hist = hist.rename(columns={TIME_COL: "ds", TARGET_COL: "y"})
    return hist


def _forecast_one(history: pd.DataFrame) -> np.ndarray:
    fallback = np.full(FORECAST_HORIZON, float(history["y"].mean()), dtype=np.float32)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with warnings.catch_warnings(), contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            warnings.simplefilter("ignore")
            try:
                model = Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=4,
                    changepoint_prior_scale=0.01,
                    seasonality_prior_scale=1.0,
                    seasonality_mode="additive",
                )
                model.fit(history)
                future = pd.DataFrame(
                    {
                        "ds": pd.date_range(
                            start=history["ds"].iloc[-1] + pd.Timedelta(minutes=30),
                            periods=FORECAST_HORIZON,
                            freq="30min",
                        )
                    }
                )
                pred = model.predict(future)["yhat"].to_numpy(dtype=np.float32)
            except Exception:
                pred = fallback
    if not np.all(np.isfinite(pred)):
        pred = fallback
    return pred


def main() -> None:
    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("prophet").disabled = True

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
