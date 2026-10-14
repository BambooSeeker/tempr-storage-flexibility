# Release validation — 13 October 2026

## Public workflow

A new Python 3.12 environment was installed from requirements-inference.txt on Windows.
Dependency consistency passed. All nine unit checks passed. Fresh inference loaded the
original LoRA adapter and pinned Chronos-2 revision, then produced 672 half-hour predictions.
The CPU BF16 run with four threads took 33.70 seconds after model caching. The measurement
includes the script's loading and prediction workflow; installation and initial download are separate.

| Window | Points | MAE (CNY/MWh) | RMSE (CNY/MWh) | MAPE (%) |
|---|---:|---:|---:|---:|
| Fig. 2 | 336 | 6.453938 | 8.553860 | 2.012103 |
| Fig. 3 | 336 | 6.833807 | 8.837949 | 2.025240 |

These window-specific live results are distinct from the archived 80-day manuscript evaluation.
The one-day market example consumed the fresh forecast, solved the candidate menu, and
retained the reference (k=1) on 5 January. Its scope is planning and causal service replay.
CPU/GPU rounding can differ; the original market forecasts and adapter remain unchanged.
The exact clean-environment package versions are recorded in docs/validated-inference-freeze.txt.

## Existing forecast benchmark correction

Five comparators were retrained with target-end-purged partitions and training-only scaling:
LightGBM, SVR, residual LSTM, residual Transformer, and residual PatchTST. Training/validation/test
counts are 13,604/1,470/3,793. Daily evaluation retains 80 days and 3,840 targets. Aggregate metrics
and measured preparation times are in config/corrected_forecast_metrics.csv and
config/corrected_timing.json. The original four-configuration comparison is retained.

## Market and record checks

The main economic input remains the original frozen LoRA forecast. Previous record-level
checks reproduced the 2,475.258638 CNY joint gain, 60% accepted-day budget reduction, and
80/80 service days for the main arrangements. This release makes no new full-period AC run.
The release includes portable workflows for authorized holders of the complete study records.
Public examples cover 14 forecast days and one planning day; the full market record bundle
requires separate access. The supplied GitHub workflow has not been executed remotely.
