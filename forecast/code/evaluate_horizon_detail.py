# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared_data import FORECAST_HORIZON, OUTPUT_DIR


DAILY_START_TIME = time(0, 30)


def _matrix(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return df[[f"{prefix}_t+{i+1}" for i in range(FORECAST_HORIZON)]].to_numpy(dtype=np.float32)


def _mae_by_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(y_true - y_pred), axis=0)


def _rmse_by_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def main() -> None:
    y_df = pd.read_csv(OUTPUT_DIR / "y_test.csv", parse_dates=["forecast_start_time"])
    pred_df = pd.read_csv(OUTPUT_DIR / "pred_LightGBM.csv", parse_dates=["forecast_start_time"])
    baseline_df = pd.read_csv(OUTPUT_DIR / "baseline_previous_day.csv", parse_dates=["forecast_start_time"])

    if not y_df["forecast_start_time"].equals(pred_df["forecast_start_time"]):
        raise ValueError("LightGBM prediction times do not match y_test times.")
    if not y_df["forecast_start_time"].equals(baseline_df["forecast_start_time"]):
        raise ValueError("Baseline times do not match y_test times.")

    y_all = _matrix(y_df, "target")
    pred_all = _matrix(pred_df, "pred")
    base_all = _matrix(baseline_df, "baseline")

    daily_mask = y_df["forecast_start_time"].dt.time == DAILY_START_TIME
    y_daily = y_all[daily_mask]
    pred_daily = pred_all[daily_mask]
    base_daily = base_all[daily_mask]

    horizons = np.arange(1, FORECAST_HORIZON + 1)
    rolling_lgb_mae = _mae_by_horizon(y_all, pred_all)
    rolling_base_mae = _mae_by_horizon(y_all, base_all)
    daily_lgb_mae = _mae_by_horizon(y_daily, pred_daily)
    daily_base_mae = _mae_by_horizon(y_daily, base_daily)

    detail = pd.DataFrame(
        {
            "horizon": horizons,
            "rolling_lightgbm_mae": rolling_lgb_mae,
            "rolling_baseline_mae": rolling_base_mae,
            "rolling_skill_score": 1.0 - rolling_lgb_mae / rolling_base_mae,
            "daily_lightgbm_mae": daily_lgb_mae,
            "daily_baseline_mae": daily_base_mae,
            "daily_skill_score": 1.0 - daily_lgb_mae / daily_base_mae,
            "rolling_lightgbm_rmse": _rmse_by_horizon(y_all, pred_all),
            "rolling_baseline_rmse": _rmse_by_horizon(y_all, base_all),
            "daily_lightgbm_rmse": _rmse_by_horizon(y_daily, pred_daily),
            "daily_baseline_rmse": _rmse_by_horizon(y_daily, base_daily),
        }
    )
    out_csv = OUTPUT_DIR / "horizon_detail_lightgbm.csv"
    detail.to_csv(out_csv, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)

    ax = axes[0]
    ax.plot(horizons, rolling_lgb_mae, marker="o", label="LightGBM rolling", color="#1f77b4")
    ax.plot(horizons, rolling_base_mae, marker="o", label="Baseline rolling", color="#ff7f0e")
    ax.set_title("Rolling evaluation: MAE by forecast horizon")
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    ax.plot(horizons, daily_lgb_mae, marker="o", label="LightGBM daily 00:30", color="#1f77b4")
    ax.plot(horizons, daily_base_mae, marker="o", label="Baseline daily 00:30", color="#ff7f0e")
    ax.set_title("Daily 00:30 non-overlapping evaluation: MAE by forecast horizon")
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.25)
    ax.legend()

    out_png = OUTPUT_DIR / "horizon_mae_lightgbm.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    print(out_csv)
    print(out_png)
    print("Daily horizon MAE head:")
    print(detail[["horizon", "daily_lightgbm_mae", "daily_baseline_mae", "daily_skill_score"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
