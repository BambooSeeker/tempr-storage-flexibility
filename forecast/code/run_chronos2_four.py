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
from chronos import Chronos2Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

from run_chronos_zero import _cols
from shared_data import FORECAST_HORIZON, OUTPUT_DIR, TARGET_COL, TIME_COL, build_dataset


DAILY_START_TIME = dtime(0, 30)
CONTEXT_LEN = 2048
MODEL_ID = os.environ.get("CHRONOS2_MODEL_ID", "amazon/chronos-2")
BATCH_SIZE = 100

ABS_EXOG_FEATURES = [
    "demand",
    "supply",
    "bidding_space",
    "renewable_penetration",
    "coal_gas_nuclear_pred",
    "photo_gene_total_pred",
    "wind_gene_total_pred",
]

DELTA_EXOG_BASE_FEATURES = [
    "demand",
    "supply",
    "bidding_space",
    "renewable_penetration",
    "coal_gas_nuclear_pred",
]


VARIANTS = [
    {
        "model": "Chronos2_Direct",
        "target_col": TARGET_COL,
        "exog_mode": "none",
        "final_mode": "direct",
    },
    {
        "model": "Chronos2_Residual",
        "target_col": "_price_residual48",
        "exog_mode": "none",
        "final_mode": "residual_plus_baseline",
    },
    {
        "model": "Chronos2_ExogDirect",
        "target_col": TARGET_COL,
        "exog_mode": "absolute",
        "final_mode": "direct",
    },
    {
        "model": "Chronos2_ExogResidual",
        "target_col": "_price_residual48",
        "exog_mode": "delta",
        "final_mode": "residual_plus_baseline",
    },
]


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bidding_space"] = (
        out["load_day_ahead_pred"]
        - out["elec_exter_plan"]
        - out["elec_fix_out_plan"]
        - out["energy_hydro_renewable"]
        - out["photo_gene_total_pred"]
        - out["wind_gene_total_pred"]
    )
    out["coal_gas_nuclear_pred"] = (
        out["elec_gene_total_pred"]
        - out["photo_gene_total_pred"]
        - out["wind_gene_total_pred"]
        - out["water_gene_total_pred"]
    )
    out["demand"] = out["load_day_ahead_pred"] - out["elec_exter_plan"]
    out["supply"] = out["elec_gene_total_pred"]
    renewable = out["wind_gene_total_pred"] + out["photo_gene_total_pred"]
    out["renewable_penetration"] = renewable / (out["supply"] + 1e-6)

    out["_price_lag48"] = out[TARGET_COL].shift(FORECAST_HORIZON)
    out["_price_residual48"] = out[TARGET_COL] - out["_price_lag48"]

    for name in DELTA_EXOG_BASE_FEATURES:
        out[f"{name}_delta48"] = out[name] - out[name].shift(FORECAST_HORIZON)

    return out


def load_pipeline(model_id: str, device: torch.device) -> tuple[Chronos2Pipeline, float]:
    kwargs = {"device_map": str(device)}
    if device.type == "cuda":
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.float32
    start = time.time()
    pipeline = Chronos2Pipeline.from_pretrained(model_id, **kwargs)
    return pipeline, time.time() - start


def _history_slice(df: pd.DataFrame, forecast_start_time: pd.Timestamp, context_len: int) -> tuple[int, int]:
    times = df[TIME_COL].to_numpy()
    idx = int(np.searchsorted(times, np.datetime64(forecast_start_time)))
    start = max(FORECAST_HORIZON, idx - context_len)
    return start, idx


def _covariate_names(exog_mode: str) -> list[str]:
    if exog_mode == "none":
        return []
    if exog_mode == "absolute":
        return ABS_EXOG_FEATURES
    if exog_mode == "delta":
        return [f"{name}_delta48" for name in DELTA_EXOG_BASE_FEATURES]
    raise ValueError(f"Unknown exog mode: {exog_mode}")


def make_tasks(
    df: pd.DataFrame,
    daily_times: pd.Series,
    target_col: str,
    exog_mode: str,
    context_len: int,
) -> list[dict[str, np.ndarray | dict[str, np.ndarray]]]:
    tasks = []
    cov_names = _covariate_names(exog_mode)

    for t in daily_times:
        start, idx = _history_slice(df, pd.Timestamp(t), context_len)
        hist = df.iloc[start:idx].copy()
        future = df.iloc[idx : idx + FORECAST_HORIZON].copy()

        if len(future) != FORECAST_HORIZON:
            raise ValueError(f"Too little future covariate window after {t}.")

        target = hist[target_col].to_numpy(dtype=np.float32)
        finite = np.isfinite(target)
        if int(finite.sum()) < FORECAST_HORIZON:
            raise ValueError(f"Too little finite target history before {t}.")
        if not bool(finite.all()):
            hist = hist.loc[finite].reset_index(drop=True)
            target = target[finite]

        task: dict[str, np.ndarray | dict[str, np.ndarray]] = {"target": target.astype(np.float32)}
        if cov_names:
            past_covariates = {}
            future_covariates = {}
            for name in cov_names:
                past_cov = hist[name].to_numpy(dtype=np.float32)
                future_cov = future[name].to_numpy(dtype=np.float32)
                if not np.all(np.isfinite(past_cov)) or not np.all(np.isfinite(future_cov)):
                    raise ValueError(f"Non-finite covariate {name} around {t}.")
                past_covariates[name] = past_cov
                future_covariates[name] = future_cov
            task["past_covariates"] = past_covariates
            task["future_covariates"] = future_covariates
        tasks.append(task)

    return tasks


def predict_point(
    pipeline: Chronos2Pipeline,
    tasks: list[dict[str, np.ndarray | dict[str, np.ndarray]]],
) -> np.ndarray:
    _, mean = pipeline.predict_quantiles(
        inputs=tasks,
        prediction_length=FORECAST_HORIZON,
        batch_size=BATCH_SIZE,
        context_length=CONTEXT_LEN,
        cross_learning=False,
        limit_prediction_length=False,
    )
    pred = np.stack([item[0].detach().cpu().numpy() for item in mean], axis=0)
    return pred.astype(np.float32)


def save_predictions(model_name: str, pred: np.ndarray, daily_times: pd.Series) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(pred, columns=_cols("pred"))
    out.insert(0, "forecast_start_time", daily_times)
    path = OUTPUT_DIR / f"daily_pred_{model_name}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def compute_and_save_metrics(
    model_name: str,
    pred: np.ndarray,
    y_daily: np.ndarray,
    baseline_daily: np.ndarray,
    daily_times: pd.Series,
    metadata: dict[str, object],
) -> dict[str, object]:
    mae = mean_absolute_error(y_daily, pred)
    rmse = mean_squared_error(y_daily, pred) ** 0.5
    baseline_mae = mean_absolute_error(y_daily, baseline_daily)
    skill = 1.0 - mae / baseline_mae if baseline_mae > 0 else 0.0
    pred_path = save_predictions(model_name, pred, daily_times)

    metrics = {
        "model": model_name,
        "eval_mode": "official_daily_00:30",
        "daily_samples": int(len(y_daily)),
        "mae": float(mae),
        "rmse": float(rmse),
        "baseline_previous_day_mae": float(baseline_mae),
        "skill_score_vs_previous_day": float(skill),
        "prediction_file": str(pred_path),
        **metadata,
    }
    metrics_path = OUTPUT_DIR / f"metrics_daily_{model_name}.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    data = build_dataset(save_outputs=True, verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = add_market_features(data.df)

    y_df = pd.DataFrame(data.y_test_raw, columns=_cols("target"))
    y_df.insert(0, "forecast_start_time", pd.to_datetime(data.test_times))
    daily_mask = y_df["forecast_start_time"].dt.time == DAILY_START_TIME
    daily_times = y_df.loc[daily_mask, "forecast_start_time"].reset_index(drop=True)
    y_daily = y_df.loc[daily_mask, _cols("target")].to_numpy(dtype=np.float32)

    baseline_df = pd.DataFrame(data.previous_day_baseline, columns=_cols("baseline"))
    baseline_daily = baseline_df.loc[daily_mask, _cols("baseline")].to_numpy(dtype=np.float32)

    pipeline, load_seconds = load_pipeline(MODEL_ID, device)
    print(f"Loaded {MODEL_ID} on {device} in {load_seconds:.2f}s")

    rows = []
    for variant in VARIANTS:
        model_name = str(variant["model"])
        target_col = str(variant["target_col"])
        exog_mode = str(variant["exog_mode"])
        final_mode = str(variant["final_mode"])
        print(f"\nRunning {model_name} ({final_mode}, exog={exog_mode})")

        tasks = make_tasks(df, daily_times, target_col, exog_mode, CONTEXT_LEN)
        start = time.time()
        raw_pred = predict_point(pipeline, tasks)
        pred_seconds = time.time() - start

        if final_mode == "residual_plus_baseline":
            pred = (baseline_daily + raw_pred).astype(np.float32)
            residual_path = OUTPUT_DIR / f"daily_residual_pred_{model_name}.csv"
            residual_out = pd.DataFrame(raw_pred, columns=[f"residual_pred_t+{i + 1}" for i in range(FORECAST_HORIZON)])
            residual_out.insert(0, "forecast_start_time", daily_times)
            residual_out.to_csv(residual_path, index=False, encoding="utf-8-sig")
        else:
            pred = raw_pred.astype(np.float32)
            residual_path = None

        metrics = compute_and_save_metrics(
            model_name=model_name,
            pred=pred,
            y_daily=y_daily,
            baseline_daily=baseline_daily,
            daily_times=daily_times,
            metadata={
                "model_id": MODEL_ID,
                "pipeline_class": "Chronos2Pipeline",
                "context_len": CONTEXT_LEN,
                "target_col": target_col,
                "exog_mode": exog_mode,
                "exog_features": _covariate_names(exog_mode),
                "final_mode": final_mode,
                "load_seconds": float(load_seconds),
                "pred_seconds": float(pred_seconds),
                "device": str(device),
                "residual_prediction_file": str(residual_path) if residual_path is not None else None,
            },
        )
        rows.append(metrics)
        print(
            f"{model_name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
            f"skill={metrics['skill_score_vs_previous_day']:.4f}, pred_seconds={pred_seconds:.2f}"
        )

    summary = pd.DataFrame(rows).sort_values("mae")
    summary_path = OUTPUT_DIR / "metrics_daily_chronos2_four.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\nChronos2 four-variant summary")
    print(summary[["model", "mae", "rmse", "skill_score_vs_previous_day", "pred_seconds"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
