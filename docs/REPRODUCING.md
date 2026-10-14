# Reproduction workflows

Use Python 3.12, and run commands from the project root. No local absolute paths are required.
Dependencies in requirements*.txt are pinned to the recorded project environment.

## 1. Public live figure examples

Install requirements-inference.txt. `python infer_figures.py` loads the original included LoRA
adapter and pinned base model, computes 14 daily forecasts, and generates Fig. 2 and Fig. 3.
`python demo.py` and `python reproduce.py --demo` call the same live inference path.
`python demo_market.py` subsequently uses the new first-day predictions for budget optimization.
See docs/LIVE_INFERENCE.md for the precision, context, data clock and comparator provenance.

## 2. Frozen full-study records

The private archive provides forecast/data, forecast/outputs, results/experiments,
data/reference_tables, data/figure_sources, data/fig1_sources and artwork.
`python reproduce.py --full` checks immutable hashes and ledger identities, recomputes metrics
and economic totals, and renders figures. `--full --no-figures` skips rendering.
Generated results are separate from frozen evidence.
Install requirements-market.txt and run `python scripts/smoke_market.py` to freshly solve
one planning day and one AC power-flow state against recorded results.

## 3. New full physical experiments

```bash
python -m pip install -r requirements-market.txt
python scripts/run_experiments.py --case main --run-dir runs/main_new
# Other choices: fixed8, pressure, unfinetuned, strong, all
```

The runner copies inputs and source into a new directory. It never overwrites frozen records.
Secondary cases use frozen reservation_v5 reference records; `--case all` first computes a
new main reference, then the four extensions. Outputs include candidate plans, selections,
15-minute AC traces and complete external-cost ledgers, with per-script logs. Runtime depends
on the machine and solver; full repeated AC runs can be long. Reruns may choose different
equivalent optima or show small numerical differences across solver/platform versions.

Settlement transfer for an accepted candidate is beta * (owner external gain + max(0,Lhat)),
with beta=0.5. Subtract it from the owner's external gain and add it to the grid saving;
the transfer cancels in joint external value. scripts/rebuild_tables.py implements this
calculation for frozen main/fixed8 records. Aggregate timing values are historical measurements.

## 4. New forecast model training

Install requirements-forecast.txt in a separate environment, obtain authorized monthly input
files, and run `python scripts/train_forecasts.py --run-dir runs/forecast_new`.
This stages code and inputs in an isolated directory and runs the original model scripts. The minimal public
example is not a training dataset. The legacy forecast/run_reproduce.py --run-models entry
requires the complete record bundle; GPU training
can yield different numerical predictions. Model scripts write forecast/outputs, so keep them
off the immutable reference copy. Chronos-2 training/inference may download model weights.
The original LoRA adapter is bundled in models/chronos2_lora; the base weights are downloaded separately.
The experimental TTM comparator has saved predictions but no training script in this release;
it is not part of the final Table II model set.

The allocation protocol locks the SHA-256 of the original LoRA forecast. A newly trained model
defines a new experiment: explicitly change the forecast hash in the working-copy protocol,
record the change, and report new outcomes separately. Do not relabel new forecasts as the
paper's frozen inputs.

## Figure 1

The approved raster is copied without redrawing. Optional PPT curve repair uses
`pip install -r requirements-artwork.txt` then `python scripts/repair_fig1_curves.py` in
the private archive. Native curve paths are generated from recorded samples; no synthetic
curve styling or generated waveform is used. The manual author's layout remains the source.

## Verification scope

Release preparation runs syntax/hash checks, data-isolation tests, the public examples,
full saved-record figure/table reconstruction, one-day MILP and one AC operating point.
It does not claim new full-GPU training, new full-80-day AC execution, or a fresh dependency
installation on every supported operating system. Check VALIDATION.md for actual outcomes.

## Corrected benchmark (1.2.0)

The corrected-baseline command in README accepts an external data directory and writes
five comparator runs to a separate output directory. The public 14-day comparator CSV
uses those runs. Full-study reconstruction must use the corrected forecast outputs and
figure sources supplied with the corresponding private record bundle. See
docs/FORECAST_PROTOCOL.md for split counts and the original model configurations.
