# Forecast protocol, version 1.2.0

## Target-window partitioning

The original 18,961 observed-target windows have 48 half-hour targets. The first validation
origin is 2025-10-23 10:00 and the first test origin is 2025-11-24 00:30. Version 1.2.0 retains
these origins and removes earlier windows whose final target reaches the following partition.
Training, validation and test contain 13,604, 1,470 and 3,793 windows. Each boundary removes
47 windows. Training and validation target intervals are disjoint; neither includes test labels.
The target scaler is fitted only on the retained training labels.

The official daily evaluation selects the 80 test origins recorded at 00:30, giving 3,840
half-hours. Timestamps denote interval ends. Three zero prices are excluded from MAPE only.
LightGBM and SVR refit on the retained training and validation windows; neural comparators
select checkpoints using the retained validation windows. All share the unchanged daily test.

The original full-series cleaning procedure is retained. For the authorized experiment records,
the evaluated 1,024/2,048-point histories in the target and seven input channels contain no
missing observations. A new dataset requires its own origin-level availability and missing-data
audit. Target-day operating variables must use the version available at the decision cutoff.

## Reported model configurations

The unadapted Chronos-2 configurations use 2,048 historical half-hours; the LoRA configuration
uses 1,024. These are the original implemented configurations. Their comparison measures
configuration performance. The four-setting comparison retains price-only direct, price-only
residual, operating-variable conditioning and the adapted operating-variable configuration.
All use the same 80 daily targets. Original market forecasts remain immutable.

## Optimization numerics

The positive-score threshold and tie tolerance are 1e-6 CNY, applied to computed candidate
scores. The MILP relative-gap request is 1e-7. The 320 archived main candidates all completed
successfully, with maximum recorded relative gap approximately 1.23e-15. Minimum sufficient
budget refers to the qualified discrete candidate menu.
