# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import time
from datetime import time as dtime
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

from run_chronos2_four import (
    ABS_EXOG_FEATURES,
    BATCH_SIZE as PRED_BATCH_SIZE,
    add_market_features,
    compute_and_save_metrics,
    make_tasks,
    predict_point,
)
from run_chronos_zero import _cols
from shared_data import FORECAST_HORIZON, OUTPUT_DIR, TARGET_COL, TIME_COL, build_dataset


MODEL_NAME = os.environ.get("CHRONOS2_LORA_MODEL_NAME", "Chronos2_ExogDirect_LoRA600")
DAILY_START_TIME = dtime(0, 30)
MODEL_ID = os.environ.get("CHRONOS2_MODEL_ID", "amazon/chronos-2")
CONTEXT_LEN = int(os.environ.get("CHRONOS2_LORA_CONTEXT_LEN", "1024"))
NUM_STEPS = int(os.environ.get("CHRONOS2_LORA_STEPS", "600"))
LEARNING_RATE = float(os.environ.get("CHRONOS2_LORA_LR", "5e-5"))
TRAIN_BATCH_SIZE = int(os.environ.get("CHRONOS2_LORA_BATCH_SIZE", "32"))
MIN_PAST = int(os.environ.get("CHRONOS2_LORA_MIN_PAST", "96"))
LORA_R = int(os.environ.get("CHRONOS2_LORA_R", "8"))
LORA_ALPHA = int(os.environ.get("CHRONOS2_LORA_ALPHA", str(LORA_R * 2)))
LORA_DROPOUT = float(os.environ.get("CHRONOS2_LORA_DROPOUT", "0.05"))


def _load_pipeline(model_id: str, device: torch.device) -> tuple[Chronos2Pipeline, float]:
    kwargs = {"device_map": str(device)}
    if device.type == "cuda":
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.float32
    start = time.time()
    pipeline = Chronos2Pipeline.from_pretrained(model_id, **kwargs)
    return pipeline, time.time() - start


def _time_index(df: pd.DataFrame, timestamp: np.datetime64 | pd.Timestamp) -> int:
    return int(np.searchsorted(df[TIME_COL].to_numpy(), np.datetime64(timestamp)))


def _task_from_slice(df: pd.DataFrame, start_idx: int, end_idx_exclusive: int) -> dict:
    part = df.iloc[start_idx:end_idx_exclusive].copy()
    target = part[TARGET_COL].to_numpy(dtype=np.float32)
    covariates = {name: part[name].to_numpy(dtype=np.float32) for name in ABS_EXOG_FEATURES}
    if not np.all(np.isfinite(target)):
        raise ValueError("Training target contains non-finite values.")
    for name, values in covariates.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Training covariate {name} contains non-finite values.")

    # Empty arrays mark these covariates as known-future for Chronos2 fine-tuning.
    return {
        "target": target,
        "past_covariates": covariates,
        "future_covariates": {name: np.asarray([], dtype=np.float32) for name in ABS_EXOG_FEATURES},
    }


def make_train_val_tasks(df: pd.DataFrame, data) -> tuple[list[dict], list[dict], dict[str, str]]:
    train_start_idx = 336
    val_start_idx = _time_index(df, pd.Timestamp(data.val_times[0]))
    test_start_idx = _time_index(df, pd.Timestamp(data.test_times[0]))
    train_end_idx = val_start_idx
    val_end_idx = test_start_idx

    train_tasks = [_task_from_slice(df, train_start_idx, train_end_idx)]
    val_tasks = [_task_from_slice(df, val_start_idx, val_end_idx)]

    split_meta = {
        "train_start": str(pd.Timestamp(df.iloc[train_start_idx][TIME_COL])),
        "train_end": str(pd.Timestamp(df.iloc[train_end_idx - 1][TIME_COL])),
        "val_start": str(pd.Timestamp(df.iloc[val_start_idx][TIME_COL])),
        "val_end": str(pd.Timestamp(df.iloc[val_end_idx - 1][TIME_COL])),
    }
    return train_tasks, val_tasks, split_meta


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

    train_tasks, val_tasks, split_meta = make_train_val_tasks(df, data)

    pipeline, load_seconds = _load_pipeline(MODEL_ID, device)
    print(f"Loaded {MODEL_ID} on {device} in {load_seconds:.2f}s")
    print(
        f"Fine-tuning {MODEL_NAME}: steps={NUM_STEPS}, lr={LEARNING_RATE}, "
        f"batch={TRAIN_BATCH_SIZE}, context={CONTEXT_LEN}, "
        f"lora_r={LORA_R}, lora_alpha={LORA_ALPHA}, lora_dropout={LORA_DROPOUT}"
    )

    output_dir = OUTPUT_DIR / "chronos2_lora_ckpts" / MODEL_NAME
    start = time.time()
    ft_pipeline = pipeline.fit(
        inputs=train_tasks,
        validation_inputs=val_tasks,
        prediction_length=FORECAST_HORIZON,
        finetune_mode="lora",
        lora_config={
            "r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": [
                "self_attention.q",
                "self_attention.v",
                "self_attention.k",
                "self_attention.o",
                "output_patch_embedding.output_layer",
            ],
        },
        context_length=CONTEXT_LEN,
        learning_rate=LEARNING_RATE,
        num_steps=NUM_STEPS,
        batch_size=TRAIN_BATCH_SIZE,
        min_past=MIN_PAST,
        output_dir=output_dir,
        finetuned_ckpt_name="final",
        remove_printer_callback=True,
        logging_steps=50,
        eval_steps=100,
        save_steps=100,
    )
    train_seconds = time.time() - start

    tasks = make_tasks(df, daily_times, TARGET_COL, "absolute", CONTEXT_LEN)
    start = time.time()
    pred = predict_point(ft_pipeline, tasks)
    pred_seconds = time.time() - start

    metrics = compute_and_save_metrics(
        model_name=MODEL_NAME,
        pred=pred,
        y_daily=y_daily,
        baseline_daily=baseline_daily,
        daily_times=daily_times,
        metadata={
            "model_id": MODEL_ID,
            "pipeline_class": "Chronos2Pipeline",
            "finetune_mode": "lora",
            "context_len": CONTEXT_LEN,
            "num_steps": NUM_STEPS,
            "learning_rate": LEARNING_RATE,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "prediction_batch_size": PRED_BATCH_SIZE,
            "target_col": TARGET_COL,
            "exog_mode": "absolute",
            "exog_features": ABS_EXOG_FEATURES,
            "final_mode": "direct",
            "load_seconds": float(load_seconds),
            "train_seconds": float(train_seconds),
            "pred_seconds": float(pred_seconds),
            "device": str(device),
            "checkpoint_dir": str(output_dir / "final"),
            **split_meta,
        },
    )

    print(f"{MODEL_NAME} done")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  skill: {metrics['skill_score_vs_previous_day']:.4f}")
    print(f"  load seconds: {load_seconds:.2f}")
    print(f"  train seconds: {train_seconds:.2f}")
    print(f"  pred seconds: {pred_seconds:.2f}")
    print(f"  checkpoint: {output_dir / 'final'}")


if __name__ == "__main__":
    main()
