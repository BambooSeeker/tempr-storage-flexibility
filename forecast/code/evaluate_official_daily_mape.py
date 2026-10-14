# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from datetime import time

import numpy as np
import pandas as pd

from shared_data import FORECAST_HORIZON, OUTPUT_DIR, build_dataset


DAILY_START_TIME = time(0, 30)


def _cols(prefix: str) -> list[str]:
    return [f"{prefix}_t+{i + 1}" for i in range(FORECAST_HORIZON)]


def _values(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return df[_cols(prefix)].to_numpy(dtype=np.float64)


def _mape_nonzero(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    mask = y_true != 0
    if int(mask.sum()) == 0:
        mape = float("nan")
    else:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    return {
        "daily_mape_percent": mape,
        "nonzero_points_used": int(mask.sum()),
        "zero_target_points_excluded": int((~mask).sum()),
    }


def main() -> None:
    if not (OUTPUT_DIR / "y_test.csv").exists() or not (OUTPUT_DIR / "baseline_previous_day.csv").exists():
        build_dataset(save_outputs=True, verbose=True)

    y_df = pd.read_csv(OUTPUT_DIR / "y_test.csv", parse_dates=["forecast_start_time"])
    baseline_df = pd.read_csv(OUTPUT_DIR / "baseline_previous_day.csv", parse_dates=["forecast_start_time"])
    daily_mask = y_df["forecast_start_time"].dt.time == DAILY_START_TIME

    y_daily_df = y_df.loc[daily_mask].reset_index(drop=True)
    baseline_daily_df = baseline_df.loc[daily_mask].reset_index(drop=True)
    y_daily = _values(y_daily_df, "target")
    baseline_daily = _values(baseline_daily_df, "baseline")
    daily_times = y_daily_df["forecast_start_time"].astype(str).to_numpy()

    rows = [
        {
            "model": "PreviousDayBaseline",
            "daily_samples": int(len(y_daily)),
            **_mape_nonzero(y_daily, baseline_daily),
        }
    ]

    for pred_path in sorted(OUTPUT_DIR.glob("pred_*.csv")):
        model = pred_path.stem.replace("pred_", "")
        pred_df = pd.read_csv(pred_path, parse_dates=["forecast_start_time"])
        if not y_df["forecast_start_time"].equals(pred_df["forecast_start_time"]):
            continue
        pred_daily = _values(pred_df.loc[daily_mask].reset_index(drop=True), "pred")
        rows.append({"model": model, "daily_samples": int(len(y_daily)), **_mape_nonzero(y_daily, pred_daily)})

    for pred_path in sorted(OUTPUT_DIR.glob("daily_pred_*.csv")):
        model = pred_path.stem.replace("daily_pred_", "")
        pred_df = pd.read_csv(pred_path, parse_dates=["forecast_start_time"])
        pred_times = pred_df["forecast_start_time"].astype(str).to_numpy()
        if len(pred_times) != len(daily_times) or not np.array_equal(pred_times, daily_times):
            raise ValueError(f"{pred_path.name} forecast_start_time does not match official daily y_test rows.")
        pred = _values(pred_df, "pred")
        rows.append({"model": model, "daily_samples": int(len(y_daily)), **_mape_nonzero(y_daily, pred)})

    summary = pd.DataFrame(rows).sort_values("daily_mape_percent")
    out_csv = OUTPUT_DIR / "metrics_official_daily_mape.csv"
    out_json = OUTPUT_DIR / "metrics_official_daily_mape.json"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Official daily MAPE evaluation")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
