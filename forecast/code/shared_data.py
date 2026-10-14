# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from chronological_split import purged_partition_indices

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("TEMPR_DATA_DIR", ROOT_DIR / "data"))
OUTPUT_DIR = Path(os.environ.get("TEMPR_OUTPUT_DIR", ROOT_DIR / "outputs"))

TARGET_UNIT = "YQFDC-01"
TARGET_COL = "price_day_ahead"
TIME_COL = "time"
CODE_COL = "code"
TARGET_OBSERVED_COL = "_target_observed"

SEQ_LEN = 192
FORECAST_HORIZON = 48
TRAIN_RATIO = 0.8
VAL_RATIO_IN_TRAIN = 0.1


@dataclass
class TargetScaler:
    mean: float | None = None
    std: float | None = None

    def fit(self, y_train: np.ndarray) -> "TargetScaler":
        self.mean = float(np.mean(y_train))
        self.std = float(np.std(y_train) + 1e-8)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetScaler must be fitted before transform().")
        return ((y - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetScaler must be fitted before inverse_transform().")
        return (y_scaled * self.std + self.mean).astype(np.float32)


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    feature_names: list[str]
    X_flat_train: np.ndarray
    X_flat_val: np.ndarray
    X_flat_test: np.ndarray
    X_seq_train: np.ndarray
    X_seq_val: np.ndarray
    X_seq_test: np.ndarray
    y_train_raw: np.ndarray
    y_val_raw: np.ndarray
    y_test_raw: np.ndarray
    y_train_scaled: np.ndarray
    y_val_scaled: np.ndarray
    y_test_scaled: np.ndarray
    train_times: np.ndarray
    val_times: np.ndarray
    test_times: np.ndarray
    previous_day_baseline: np.ndarray
    scaler: TargetScaler
    n_samples_before_invalid_target_filter: int
    n_samples_removed_for_missing_target_window: int


def load_and_clean_all_data(data_dir: Path = DATA_DIR, unit_id: str = TARGET_UNIT) -> pd.DataFrame:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    df = pd.concat([pd.read_csv(f, parse_dates=[TIME_COL]) for f in files], ignore_index=True)
    if CODE_COL not in df.columns:
        raise KeyError(f"Missing required column: {CODE_COL}")

    df = df[df[CODE_COL] == unit_id].reset_index(drop=True)
    df = df.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL], keep="last")
    df = df.set_index(TIME_COL).asfreq("30min").reset_index()
    df[TARGET_OBSERVED_COL] = df[TARGET_COL].notna()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
    df = df.ffill().bfill()
    return df


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = df.copy()

    df["bidding_space"] = (
        df["load_day_ahead_pred"]
        - df["elec_exter_plan"]
        - df["elec_fix_out_plan"]
        - df["energy_hydro_renewable"]
        - df["photo_gene_total_pred"]
        - df["wind_gene_total_pred"]
    )
    df["coal_gas_nuclear_pred"] = (
        df["elec_gene_total_pred"]
        - df["photo_gene_total_pred"]
        - df["wind_gene_total_pred"]
        - df["water_gene_total_pred"]
    )
    df["demand"] = df["load_day_ahead_pred"] - df["elec_exter_plan"]
    df["supply"] = df["elec_gene_total_pred"]
    renewable = df["wind_gene_total_pred"] + df["photo_gene_total_pred"]
    df["renewable_penetration"] = renewable / (df["supply"] + 1e-6)

    synchronous_vars = [
        "coal_gas_nuclear_pred",
        "demand",
        "supply",
        "renewable_penetration",
        "photo_gene_total_pred",
        "wind_gene_total_pred",
        "bidding_space",
    ]
    lag48_vars = [
        "bidding_space",
        "price_day_ahead_energy",
        "price_day_ahead_cong",
        "elec_day_ahead_all",
        "price_day_ahead_load",
        "coal_gas_nuclear_pred",
        "demand",
        "supply",
        "renewable_penetration",
    ]
    lag49_vars = ["coal_gas_nuclear_pred", "demand", "supply", "bidding_space"]
    lag96_vars = ["coal_gas_nuclear_real", "load_real_neat", "coal_gas_nuclear_pred", "demand", "supply", "bidding_space"]
    lag144_vars = ["price_real", "price_real_load", "coal_gas_nuclear_pred", "demand", "supply", "bidding_space"]
    lag192_vars = ["price_real_energy", "price_real_cong", "elec_real_all", "coal_gas_nuclear_pred", "demand", "supply", "bidding_space"]

    for lag, cols in [(48, lag48_vars), (49, lag49_vars), (96, lag96_vars), (144, lag144_vars), (192, lag192_vars)]:
        for col in cols:
            if col in df.columns:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

    df["price_day_ahead_lag48"] = df[TARGET_COL].shift(48)
    df["price_day_ahead_lag49"] = df[TARGET_COL].shift(49)
    df["price_day_ahead_lag96"] = df[TARGET_COL].shift(96)
    df["price_day_ahead_lag144"] = df[TARGET_COL].shift(144)
    df["price_day_ahead_lag336"] = df[TARGET_COL].shift(336)
    df["price_day_ahead_rollmean24h"] = df[TARGET_COL].shift(48).rolling(48, min_periods=1).mean()
    df["price_day_ahead_rollstd24h"] = df[TARGET_COL].shift(48).rolling(48, min_periods=1).std()
    df["price_day_ahead_rollmax24h"] = df[TARGET_COL].shift(48).rolling(48, min_periods=1).max()
    df["price_day_ahead_rollmin24h"] = df[TARGET_COL].shift(48).rolling(48, min_periods=1).min()
    df["price_day_ahead_peakvalley24h"] = df["price_day_ahead_rollmax24h"] - df["price_day_ahead_rollmin24h"]

    feature_names: list[str] = []
    feature_names.extend([c for c in synchronous_vars if c in df.columns])
    for lag, cols in [(48, lag48_vars), (49, lag49_vars), (96, lag96_vars), (144, lag144_vars), (192, lag192_vars)]:
        feature_names.extend([f"{c}_lag{lag}" for c in cols if f"{c}_lag{lag}" in df.columns])

    price_lag_cols = [
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
    ]
    feature_names.extend([c for c in price_lag_cols if c in df.columns])

    df_valid = df.iloc[336:].copy()
    df_valid = df_valid.dropna(subset=feature_names + [TARGET_COL]).reset_index(drop=True)

    X = df_valid[feature_names].to_numpy(dtype=np.float32)
    y = df_valid[TARGET_COL].to_numpy(dtype=np.float32)
    times = df_valid[TIME_COL].to_numpy()
    target_observed = df_valid[TARGET_OBSERVED_COL].to_numpy(dtype=bool)
    return X, y, times, target_observed, feature_names


def create_tabular_samples(
    X: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    target_observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create tabular samples from the known exogenous target window.

    For industrial day-ahead forecasting, the market/load/generation forecasts
    in the target window are known inputs, so tabular models use
    X[t:t+48].reshape(-1) -> y[t:t+48].
    """
    n = len(X)
    if n < SEQ_LEN + FORECAST_HORIZON:
        raise ValueError("Not enough rows to create tabular samples.")

    n_samples = n - SEQ_LEN - FORECAST_HORIZON + 1
    X_flat = np.zeros((n_samples, FORECAST_HORIZON * X.shape[1]), dtype=np.float32)
    y_multi = np.zeros((n_samples, FORECAST_HORIZON), dtype=np.float32)
    valid_target = np.zeros(n_samples, dtype=bool)
    sample_times = []

    for i in range(n_samples):
        forecast_start = SEQ_LEN + i
        X_flat[i] = X[forecast_start : forecast_start + FORECAST_HORIZON].reshape(-1)
        y_multi[i] = y[forecast_start : forecast_start + FORECAST_HORIZON]
        valid_target[i] = bool(np.all(target_observed[forecast_start : forecast_start + FORECAST_HORIZON]))
        sample_times.append(times[forecast_start])

    return X_flat, y_multi, np.asarray(sample_times), valid_target


def create_sequence_samples(
    X: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    target_observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    if n < SEQ_LEN + FORECAST_HORIZON:
        raise ValueError("Not enough rows to create sequence samples.")

    n_samples = n - SEQ_LEN - FORECAST_HORIZON + 1
    input_len = SEQ_LEN + FORECAST_HORIZON
    X_seq = np.zeros((n_samples, input_len, X.shape[1]), dtype=np.float32)
    y_multi = np.zeros((n_samples, FORECAST_HORIZON), dtype=np.float32)
    valid_target = np.zeros(n_samples, dtype=bool)
    sample_times = []

    for i in range(n_samples):
        forecast_start = SEQ_LEN + i
        X_seq[i] = X[i : forecast_start + FORECAST_HORIZON]
        y_multi[i] = y[forecast_start : forecast_start + FORECAST_HORIZON]
        valid_target[i] = bool(np.all(target_observed[forecast_start : forecast_start + FORECAST_HORIZON]))
        sample_times.append(times[forecast_start])

    return X_seq, y_multi, np.asarray(sample_times), valid_target


def make_previous_day_baseline(df: pd.DataFrame, test_times: np.ndarray) -> np.ndarray:
    full_times = df[TIME_COL].to_numpy()
    full_prices = df[TARGET_COL].to_numpy(dtype=np.float32)
    rows = []
    for t in test_times:
        idx = np.searchsorted(full_times, np.datetime64(t))
        hist = full_prices[max(0, idx - FORECAST_HORIZON) : idx]
        if len(hist) < FORECAST_HORIZON:
            hist = np.pad(hist, (FORECAST_HORIZON - len(hist), 0), mode="edge")
        rows.append(hist[-FORECAST_HORIZON:])
    return np.asarray(rows, dtype=np.float32)


def _assert_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        bad = int(np.sum(~np.isfinite(arr)))
        raise ValueError(f"{name} contains {bad} non-finite values.")


def build_dataset(save_outputs: bool = True, verbose: bool = True) -> DatasetBundle:
    df = load_and_clean_all_data()
    X, y, times, target_observed, feature_names = build_features(df)
    X_flat, y_flat, t_flat, valid_flat = create_tabular_samples(X, y, times, target_observed)
    X_seq, y_seq, t_seq, valid_seq = create_sequence_samples(X, y, times, target_observed)

    if not np.array_equal(t_flat, t_seq):
        raise AssertionError("Tabular and sequence sample times are not aligned.")
    if not np.allclose(y_flat, y_seq):
        raise AssertionError("Tabular and sequence targets are not aligned.")
    if not np.array_equal(valid_flat, valid_seq):
        raise AssertionError("Tabular and sequence target-valid masks are not aligned.")

    n_before_filter = len(X_flat)
    valid_target_mask = valid_flat
    X_flat = X_flat[valid_target_mask]
    y_flat = y_flat[valid_target_mask]
    t_flat = t_flat[valid_target_mask]
    X_seq = X_seq[valid_target_mask]
    y_seq = y_seq[valid_target_mask]
    t_seq = t_seq[valid_target_mask]

    if not np.allclose(X_flat[0].reshape(FORECAST_HORIZON, X.shape[1]), X[SEQ_LEN : SEQ_LEN + FORECAST_HORIZON]):
        raise AssertionError("Tabular sample 0 is not X[t:t+48].")
    if not np.allclose(X_seq[0], X[: SEQ_LEN + FORECAST_HORIZON]):
        raise AssertionError("Sequence sample 0 is not X[t-192:t+48].")

    n_samples = len(X_flat)
    split_test = int(n_samples * TRAIN_RATIO)
    split_val = int(split_test * (1.0 - VAL_RATIO_IN_TRAIN))

    # Preserve the original test origins, but purge every earlier window whose
    # final target reaches the next partition. Fit preprocessing on training only.
    train_slice, val_slice, test_slice = purged_partition_indices(
        t_flat, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO_IN_TRAIN
    )
    scaler = TargetScaler().fit(y_flat[train_slice])

    previous_day_baseline = make_previous_day_baseline(df, t_flat[test_slice])

    bundle = DatasetBundle(
        df=df,
        feature_names=feature_names,
        X_flat_train=X_flat[train_slice],
        X_flat_val=X_flat[val_slice],
        X_flat_test=X_flat[test_slice],
        X_seq_train=X_seq[train_slice],
        X_seq_val=X_seq[val_slice],
        X_seq_test=X_seq[test_slice],
        y_train_raw=y_flat[train_slice],
        y_val_raw=y_flat[val_slice],
        y_test_raw=y_flat[test_slice],
        y_train_scaled=scaler.transform(y_flat[train_slice]),
        y_val_scaled=scaler.transform(y_flat[val_slice]),
        y_test_scaled=scaler.transform(y_flat[test_slice]),
        train_times=t_flat[train_slice],
        val_times=t_flat[val_slice],
        test_times=t_flat[test_slice],
        previous_day_baseline=previous_day_baseline,
        scaler=scaler,
        n_samples_before_invalid_target_filter=n_before_filter,
        n_samples_removed_for_missing_target_window=n_before_filter - n_samples,
    )

    for name in [
        "X_flat_train",
        "X_flat_val",
        "X_flat_test",
        "X_seq_train",
        "X_seq_val",
        "X_seq_test",
        "y_train_raw",
        "y_val_raw",
        "y_test_raw",
        "previous_day_baseline",
    ]:
        _assert_finite(name, getattr(bundle, name))

    if save_outputs:
        save_common_outputs(bundle)

    if verbose:
        print_dataset_report(bundle, len(df), len(X))

    return bundle


def save_common_outputs(bundle: DatasetBundle) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_test_df = pd.DataFrame(bundle.y_test_raw, columns=[f"target_t+{i+1}" for i in range(FORECAST_HORIZON)])
    y_test_df.insert(0, "forecast_start_time", bundle.test_times)
    y_test_df.to_csv(OUTPUT_DIR / "y_test.csv", index=False, encoding="utf-8-sig")

    baseline_df = pd.DataFrame(bundle.previous_day_baseline, columns=[f"baseline_t+{i+1}" for i in range(FORECAST_HORIZON)])
    baseline_df.insert(0, "forecast_start_time", bundle.test_times)
    baseline_df.to_csv(OUTPUT_DIR / "baseline_previous_day.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"feature": bundle.feature_names}).to_csv(OUTPUT_DIR / "feature_names.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "data_dir": str(DATA_DIR),
        "target_unit": TARGET_UNIT,
        "target_col": TARGET_COL,
        "seq_len": SEQ_LEN,
        "forecast_horizon": FORECAST_HORIZON,
        "train_ratio": TRAIN_RATIO,
        "val_ratio_in_train": VAL_RATIO_IN_TRAIN,
        "n_train": int(len(bundle.y_train_raw)),
        "split_protocol": "target-end-purged-v3",
        "target_scaler_fit": "training targets only",
        "train_last_target": str(pd.Timestamp(bundle.train_times[-1]) + pd.Timedelta(minutes=30*(FORECAST_HORIZON-1))),
        "validation_first_origin": str(pd.Timestamp(bundle.val_times[0])),
        "validation_last_target": str(pd.Timestamp(bundle.val_times[-1]) + pd.Timedelta(minutes=30*(FORECAST_HORIZON-1))),
        "n_val": int(len(bundle.y_val_raw)),
        "n_test": int(len(bundle.y_test_raw)),
        "n_samples_before_invalid_target_filter": int(bundle.n_samples_before_invalid_target_filter),
        "n_samples_after_invalid_target_filter": int(len(bundle.y_train_raw) + len(bundle.y_val_raw) + len(bundle.y_test_raw)),
        "n_samples_removed_for_missing_target_window": int(bundle.n_samples_removed_for_missing_target_window),
        "n_features": int(len(bundle.feature_names)),
        "X_flat_train_shape": list(bundle.X_flat_train.shape),
        "X_seq_train_shape": list(bundle.X_seq_train.shape),
        "test_start": str(pd.Timestamp(bundle.test_times[0])),
        "test_end": str(pd.Timestamp(bundle.test_times[-1])),
        "target_scaler_mean": bundle.scaler.mean,
        "target_scaler_std": bundle.scaler.std,
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def print_dataset_report(bundle: DatasetBundle, raw_rows: int, feature_rows: int) -> None:
    print("Dataset report")
    print(f"  raw rows: {raw_rows}")
    print(f"  feature rows after lag/dropna: {feature_rows}")
    print(f"  features: {len(bundle.feature_names)}")
    print(
        "  invalid target-window samples removed: "
        f"{bundle.n_samples_removed_for_missing_target_window} / {bundle.n_samples_before_invalid_target_filter}"
    )
    print(f"  tabular X: train {bundle.X_flat_train.shape}, val {bundle.X_flat_val.shape}, test {bundle.X_flat_test.shape}")
    print(f"  sequence X: train {bundle.X_seq_train.shape}, val {bundle.X_seq_val.shape}, test {bundle.X_seq_test.shape}")
    print(f"  y: train {bundle.y_train_raw.shape}, val {bundle.y_val_raw.shape}, test {bundle.y_test_raw.shape}")
    print(f"  test time: {pd.Timestamp(bundle.test_times[0])} -> {pd.Timestamp(bundle.test_times[-1])}")
    print(f"  target scaler: mean={bundle.scaler.mean:.6f}, std={bundle.scaler.std:.6f}")


if __name__ == "__main__":
    build_dataset(save_outputs=True, verbose=True)
