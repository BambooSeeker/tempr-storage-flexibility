# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import json
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


MODEL_NAME = "LSTM"
SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 80
PATIENCE = 12
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 256
FC_SIZE = 128
DROPOUT = 0.1

SELECTED_FEATURES = [
    "price_day_ahead_energy_lag48",
    "price_day_ahead_cong_lag48",
    "elec_day_ahead_all_lag48",
    "price_day_ahead_load_lag48",
    "price_day_ahead_lag48",
    "price_day_ahead_lag49",
    "price_day_ahead_lag96",
    "price_day_ahead_lag144",
    "price_day_ahead_lag336",
    "price_day_ahead_rollmean24h",
    "price_day_ahead_rollstd24h",
    "price_day_ahead_rollmax24h",
    "price_day_ahead_rollmin24h",
    "price_day_ahead_peakvalley24h",
    "demand",
    "supply",
    "bidding_space",
    "renewable_penetration",
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(HIDDEN_SIZE)
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, FC_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FC_SIZE, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -FORECAST_HORIZON:, :]
        out = self.norm(out)
        return self.head(out).squeeze(-1)


def select_features(X: np.ndarray, all_feature_names: list[str]) -> np.ndarray:
    missing = [name for name in SELECTED_FEATURES if name not in all_feature_names]
    if missing:
        raise ValueError(f"Selected LSTM features are missing: {missing}")
    indices = [all_feature_names.index(name) for name in SELECTED_FEATURES]
    return X[:, :, indices].astype(np.float32)


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
            pred = model(xb)
            losses.append(float(criterion(pred, yb).item()))
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

    X_train_raw = select_features(data.X_seq_train[:, -FORECAST_HORIZON:, :], data.feature_names)
    X_val_raw = select_features(data.X_seq_val[:, -FORECAST_HORIZON:, :], data.feature_names)
    X_test_raw = select_features(data.X_seq_test[:, -FORECAST_HORIZON:, :], data.feature_names)
    X_train, X_val, X_test, _ = scale_sequence_features(X_train_raw, X_val_raw, X_test_raw)
    y_train = data.y_train_scaled.astype(np.float32)
    y_val = data.y_val_scaled.astype(np.float32)

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    model = LSTMForecaster(input_size=X_train.shape[-1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

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
            pred = model(xb)
            loss = criterion(pred, yb)
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
    pred_scaled = predict(model, X_test, device)
    pred_seconds = time.time() - start
    pred = data.scaler.inverse_transform(pred_scaled)

    mae = mean_absolute_error(data.y_test_raw, pred)
    rmse = mean_squared_error(data.y_test_raw, pred) ** 0.5
    baseline_mae = mean_absolute_error(data.y_test_raw, data.previous_day_baseline)
    skill_score = 1.0 - mae / baseline_mae if baseline_mae > 0 else 0.0

    pred_path = save_predictions(pred, data.test_times)
    checkpoint_path = OUTPUT_DIR / f"model_{MODEL_NAME}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": int(X_train.shape[-1]),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "config": {
                "hidden_size": HIDDEN_SIZE,
                "fc_size": FC_SIZE,
                "num_layers": 1,
                "dropout": DROPOUT,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "seed": SEED,
            },
        },
        checkpoint_path,
    )
    history_path = OUTPUT_DIR / f"history_{MODEL_NAME}.csv"
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")

    metrics = {
        "model": MODEL_NAME,
        "input_mode": "sequence_X[t:t+48]_selected_core_features_horizon_aligned",
        "target_mode": "direct_price_prediction",
        "selected_features": SELECTED_FEATURES,
        "input_shape_train": list(X_train.shape),
        "input_shape_val": list(X_val.shape),
        "input_shape_test": list(X_test.shape),
        "n_features": int(X_train.shape[-1]),
        "seq_len": int(X_train.shape[1]),
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
        "checkpoint_file": str(checkpoint_path),
        "history_file": str(history_path),
    }
    metrics_path = OUTPUT_DIR / f"metrics_{MODEL_NAME}.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{MODEL_NAME} done")
    print(f"  device: {device}")
    print("  target mode: direct price prediction")
    print(f"  input shape: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
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
