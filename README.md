# Storage flexibility allocation for energy trading and network services

Research implementation accompanying **Storage Flexibility Allocation for Energy Trading and
Network Services With Time-Series Foundation Models**. Version 1.2.0.

Source: https://github.com/BambooSeeker/tempr-storage-flexibility

Versioned release: https://github.com/BambooSeeker/tempr-storage-flexibility/releases/tag/v1.2.0

The package contains forecasting, budget optimization, causal qualification, network execution,
settlement and figure code. The original LoRA adapter and two weeks of authorized example inputs
support fresh price inference. Full model training uses separately authorized market records.

## Run the public example

Use Python 3.12 in a new virtual environment.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python reproduce.py --check
python -m unittest discover -s tests -v
python infer_figures.py
python demo_market.py
```

The first inference downloads the pinned Chronos-2 base checkpoint. The included adapter is
loaded after its SHA-256 is checked. Cached runs can use `python infer_figures.py --offline`.
CPU BF16 with four threads is the default; `--precision float32` and `--device cuda` are optional.

Outputs appear in `generated/live_figures` and `generated/market_demo`. Fig. 2 and Fig. 3 each
cover seven days. LoRA predictions are freshly computed; the other model curves are archived
predictions for those same days. The market example solves four budget candidates and tests
causal service qualification for 5 January. That example retains the reference schedule.

## Reproduce the corrected forecast benchmark

Install `requirements-forecast.txt` and provide authorized monthly CSV files.

```bash
python forecast/run_corrected_baselines.py --data-dir /path/to/monthly_csv --output-dir /path/to/new_results
```

Version 1.2.0 purges multi-step target windows crossing partition boundaries and fits the target
scaler on training labels. The 80 test days and the original four forecasting configurations remain unchanged.
See `docs/FORECAST_PROTOCOL.md`, `docs/MODEL_INVENTORY.md` and `VALIDATION.md`.

## Data, results and scope

Public market inputs comprise 1,024 historical half-hours and 672 evaluation half-hours, with
the target and seven operating covariates. Full training records and the complete 80-day price
series remain in the authorized archive. Aggregate results and the protocol are supplied in
`config`. SimBench-derived network inputs are distributed under their upstream terms.

The storage study retains its original frozen LoRA forecasts. Corrected comparator training has separate provenance and preserves that market input.
Full archived-result reconstruction and fresh physical experiments are documented in
`docs/REPRODUCING.md`. The public example supports model inference and one-day planning; full
80-day reconstruction requires the complete record bundle.

Original code and the authors' adapter are MIT licensed. Chronos-2 base weights and SimBench
data retain their separate licenses. See `LICENSE`, `THIRD_PARTY_NOTICES.md`,
`DATA_AVAILABILITY.md`, and `data/DATA_TERMS.md`. Citation metadata is in `CITATION.cff`.
