# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from shared_data import FORECAST_HORIZON, OUTPUT_DIR, build_dataset
from run_lstm_residual_delta import make_delta_features


MODEL_NAME = "PatchTST_ResidualDelta"
SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE = 14
LEARNING_RATE = 8e-5
WEIGHT_DECAY = 1e-4
PATCH_LEN = 4
STRIDE = 2
D_MODEL = 64
N_HEAD = 4
N_LAYERS = 2
FF_SIZE = 160
DROPOUT = 0.1


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class PatchTSTResidualForecaster(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.n_patches = 1 + (FORECAST_HORIZON - PATCH_LEN) // STRIDE
        self.patch_proj = nn.Sequential(
            nn.Linear(PATCH_LEN * input_size, D_MODEL),
            nn.LayerNorm(D_MODEL),
            nn.GELU(),
        )
        self.positional = PositionalEncoding(D_MODEL, self.n_patches)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEAD,
            dim_feedforward=FF_SIZE,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(self.n_patches * D_MODEL),
            nn.Dropout(DROPOUT),
            nn.Linear(self.n_patches * D_MODEL, 192),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(192, FORECAST_HORIZON),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=PATCH_LEN, step=STRIDE)
        patches = patches.permute(0, 1, 3, 2).reshape(x.size(0), self.n_patches, -1)
        z = self.patch_proj(patches)
        z = self.positional(z)
        z = self.encoder(z)
        return self.head(z)


def scale_sequence_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    n_train, seq_len, n_features = X_train.shape
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_train, seq_len, n_features)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    return (
        X_train_scaled.astype(np.float32),
        X_val_scaled.astype(np.float32),
        X_test_scaled.astype(np.float32),
        scaler,
    )


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            losses.append(float(criterion(model(xb), yb).item()))
    return float(np.mean(losses))


def predict(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    loader = make_loader(X, np.zeros((len(X), FORECAST_HORIZON), dtype=np.float32), BATCH_SIZE, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def save_predictions(pred: np.ndarray, test_times: np.ndarray) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(pred, columns=[f"pred_t+{i + 1}" for i in range(FORECAST_HORIZON)])
    out.insert(0, "forecast_start_time", test_times)
    path = OUTPUT_DIR / f"pred_{MODEL_NAME}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    set_seed(SEED)
    data = build_dataset(save_outputs=True, verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_window = data.X_seq_train[:, -FORECAST_HORIZON:, :]
    val_window = data.X_seq_val[:, -FORECAST_HORIZON:, :]
    test_window = data.X_seq_test[:, -FORECAST_HORIZON:, :]

    X_train_raw, delta_feature_names = make_delta_features(train_window, data.train_times, data.feature_names)
    X_val_raw, _ = make_delta_features(val_window, data.val_times, data.feature_names)
    X_test_raw, _ = make_delta_features(test_window, data.test_times, data.feature_names)
    X_train, X_val, X_test, _ = scale_sequence_features(X_train_raw, X_val_raw, X_test_raw)

    lag48_idx = data.feature_names.index("price_day_ahead_lag48")
    train_base = train_window[:, :, lag48_idx].astype(np.float32)
    val_base = val_window[:, :, lag48_idx].astype(np.float32)
    test_base = test_window[:, :, lag48_idx].astype(np.float32)

    y_train = ((data.y_train_raw - train_base) / data.scaler.std).astype(np.float32)
    y_val = ((data.y_val_raw - val_base) / data.scaler.std).astype(np.float32)

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    model = PatchTSTResidualForecaster(input_size=X_train.shape[-1]).to(device)
    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    history = []
    start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"early stopping at epoch {epoch}, best epoch {best_epoch}")
                break

    train_seconds = time.time() - start
    if best_state is not None:
        model.load_state_dict(best_state)

    start = time.time()
    pred_delta_scaled = predict(model, X_test, device)
    pred_seconds = time.time() - start
    pred = (test_base + pred_delta_scaled * data.scaler.std).astype(np.float32)

    mae = mean_absolute_error(data.y_test_raw, pred)
    rmse = mean_squared_error(data.y_test_raw, pred) ** 0.5
    baseline_mae = mean_absolute_error(data.y_test_raw, data.previous_day_baseline)
    skill_score = 1.0 - mae / baseline_mae if baseline_mae > 0 else 0.0

    pred_path = save_predictions(pred, data.test_times)
    history_path = OUTPUT_DIR / f"history_{MODEL_NAME}.csv"
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")

    metrics = {
        "model": MODEL_NAME,
        "input_mode": "patchtst_market_delta48_price_change_level_time_features_X[t:t+48]",
        "target_mode": "predict_price_delta_y_minus_price_day_ahead_lag48",
        "patch_len": PATCH_LEN,
        "stride": STRIDE,
        "delta_feature_names": delta_feature_names,
        "input_shape_train": list(X_train.shape),
        "input_shape_val": list(X_val.shape),
        "input_shape_test": list(X_test.shape),
        "mae": float(mae),
        "rmse": float(rmse),
        "baseline_previous_day_mae": float(baseline_mae),
        "skill_score_vs_previous_day": float(skill_score),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "train_seconds": float(train_seconds),
        "pred_seconds": float(pred_seconds),
        "device": str(device),
        "prediction_file": str(pred_path),
        "history_file": str(history_path),
    }
    metrics_path = OUTPUT_DIR / f"metrics_{MODEL_NAME}.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{MODEL_NAME} done")
    print(f"  device: {device}")
    print(f"  input shape: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
    print(f"  delta features: {delta_feature_names}")
    print(f"  best epoch: {best_epoch}, best val loss: {best_val:.5f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  previous-day baseline MAE: {baseline_mae:.4f}")
    print(f"  skill score: {skill_score:.4f}")
    print(f"  train seconds: {train_seconds:.2f}")
    print(f"  pred seconds: {pred_seconds:.2f}")
    print(f"  saved: {pred_path}")


if __name__ == "__main__":
    main()
