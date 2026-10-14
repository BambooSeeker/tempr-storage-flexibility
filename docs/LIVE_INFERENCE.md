# Model and figure provenance

The adapter is the original `Chronos2_ExogDirect_LoRA600/final` checkpoint. Its sibling saved
80-day forecast has SHA-256 f57d9115b44355d23fa9d78d784b8eae0223f26d115ffde8ce1f84f5fedf4989,
matching the frozen paper protocol. Adapter SHA-256 and the pinned base model revision are
in config/figure_demo.json. Inference checks the adapter hash before loading.

Input feature engineering matches run_chronos2_four.py. The original loader cleans the private
raw records before the minimum public window is extracted. No new day selection or retraining
is performed. The original Fig. 2 source was called Forecast_month despite being seven days;
its actual window is 12–18 January. Fig. 3 is 5–11 January.

The training API copied the BF16-loaded base weights into FP32 before attaching LoRA.
Loading recreates that state and then applies BF16 inference, preserving the adapter as
separate modules. It does not merge and round the adapter into the base weights.
The task history has 1,024 points; the original predict_point helper passed context_length=2048
but supplied only that 1,024-point task. The released inference preserves these arguments.

For each day: assemble historical target and seven past/future covariates; compute 48 points;
then reveal that day's observed price for evaluation and next-day history. Published comparator
predictions are read only after all LoRA predictions have been generated. Their provenance is
recorded in each run.json. They are never model inputs.

Model rounding and CPU/GPU kernels can yield small differences from saved paper predictions.
The release audit reports them directly, without adjusting predictions to match old curves.
The generated PNG/PDF/SVG are fresh inference figures; the original manuscript is unchanged.
