from __future__ import annotations

import argparse
import os
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("TEMPR_DATA_DIR", ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("TEMPR_OUTPUT_DIR", ROOT / "outputs"))
CODE_DIR = ROOT / "code"

EXPECTED = {
    "Chronos2_ExogDirect_LoRA600": {"mae": 16.0173, "mape": 6.9721},
    "Chronos2_ExogDirect": {"mae": 16.5993, "mape": 7.2057},
    "Chronos2_Direct": {"mae": 20.3760, "mape": 9.0551},
    "LightGBM": {"mae": 21.7355, "mape": 7.4991},
    "SVR_PriceLags": {"mae": 21.8391, "mape": 9.5659},
    "ARIMA": {"mae": 22.1382, "mape": 9.6885},
    "Prophet": {"mae": 23.3189, "mape": 9.8370},
    "LSTM_ResidualDelta": {"mae": 23.5130, "mape": 9.2143},
    "Transformer_ResidualDelta": {"mae": 22.5895, "mape": 8.5832},
    "PatchTST_ResidualDelta": {"mae": 25.0645, "mape": 9.6529},
}


MODEL_SCRIPTS = [
    "run_lightgbm.py",
    "run_svr_pricelags.py",
    "run_arima.py",
    "run_prophet.py",
    "run_lstm.py",
    "run_lstm_residual_delta.py",
    "run_transformer_direct.py",
    "run_transformer_residual_delta.py",
    "run_patchtst_direct.py",
    "run_patchtst_residual_delta.py",
    "run_chronos_zero.py",
    "run_chronos2_four.py",
    "run_chronos2_exog_lora.py",
    "evaluate_official_daily.py",
    "evaluate_official_daily_mape.py",
]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {path}")


def check_inputs() -> None:
    csv_files = sorted(DATA_DIR.glob("input_price_predict_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No monthly input CSV files found under {DATA_DIR}")
    require_file(OUTPUT_DIR / "metrics_official_daily.csv")
    require_file(OUTPUT_DIR / "metrics_official_daily_mape.csv")
    require_file(OUTPUT_DIR / "metrics_daily_chronos2_four.csv")
    print(f"[OK] Found {len(csv_files)} monthly input files.")
    print("[OK] Found required metric files.")


def make_reproduced_table() -> pd.DataFrame:
    daily = pd.read_csv(OUTPUT_DIR / "metrics_official_daily.csv")
    mape = pd.read_csv(OUTPUT_DIR / "metrics_official_daily_mape.csv")
    merged = daily.merge(
        mape[["model", "daily_mape_percent"]],
        on="model",
        how="left",
    )
    cols = [
        "model",
        "daily_samples",
        "mae",
        "rmse",
        "daily_mape_percent",
        "skill_score_vs_previous_day",
    ]
    table = merged[cols].sort_values("mae").reset_index(drop=True)
    out_path = OUTPUT_DIR / "reproduced_table1.csv"
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Wrote {out_path}")
    return table


def verify_expected(table: pd.DataFrame, tol: float = 5e-4) -> None:
    by_model = table.set_index("model")
    failed = []
    for model, metrics in EXPECTED.items():
        if model not in by_model.index:
            failed.append(f"{model}: missing")
            continue
        mae = float(by_model.loc[model, "mae"])
        mape = float(by_model.loc[model, "daily_mape_percent"])
        if abs(mae - metrics["mae"]) > tol:
            failed.append(f"{model}: MAE {mae:.4f} != {metrics['mae']:.4f}")
        if abs(mape - metrics["mape"]) > tol:
            failed.append(f"{model}: MAPE {mape:.4f} != {metrics['mape']:.4f}")
    if failed:
        raise AssertionError("Metric check failed:\n" + "\n".join(failed))
    print("[OK] Key paper metrics match expected values.")


def write_summary(table: pd.DataFrame) -> None:
    top = table.head(12).copy()
    summary = {
        "official_daily_samples": int(table["daily_samples"].max()),
        "best_model": str(table.iloc[0]["model"]),
        "best_mae": float(table.iloc[0]["mae"]),
        "best_mape_percent": float(table.iloc[0]["daily_mape_percent"]),
        "table_preview": top[
            ["model", "mae", "rmse", "daily_mape_percent"]
        ].to_dict(orient="records"),
    }
    out_path = OUTPUT_DIR / "reproduction_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Wrote {out_path}")


def run_model_scripts() -> None:
    print("[INFO] Running full model scripts. This may require GPU and Chronos/torch dependencies.")
    for script in MODEL_SCRIPTS:
        path = CODE_DIR / script
        require_file(path)
        print(f"[RUN] {script}")
        subprocess.run([sys.executable, str(path)], cwd=str(CODE_DIR), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce and verify the paper experiment outputs."
    )
    parser.add_argument(
        "--run-models",
        action="store_true",
        help="Run all model scripts before rebuilding summary metrics. Requires full ML/DL environment.",
    )
    args = parser.parse_args()

    if args.run_models:
        run_model_scripts()
    check_inputs()
    table = make_reproduced_table()
    if not args.run_models:
        verify_expected(table)
    write_summary(table)
    print("\nReproduction check finished successfully.")


if __name__ == "__main__":
    main()
