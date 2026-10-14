# Paper LoRA adapter

Original Chronos2_ExogDirect_LoRA600 final adapter: rank 8, alpha 16, dropout 0.05,
600 training steps, context 1024, attention Q/K/V/O and output patch projection.
The checkpoint's accompanying frozen prediction file matches the paper protocol SHA-256.
Training and validation data are not bundled. The adapter contains numerical model parameters.
Original adapter contributions are released under MIT; the Amazon Chronos-2 base model retains
Apache 2.0 and is downloaded separately at the pinned revision in config/figure_demo.json.

Reconstruction preserves the original BF16-rounded base weights promoted to FP32 by fit(),
and attaches the adapter without merging. Numerical comparisons with paper outputs are recorded
in the release validation report; do not infer exact equality across all devices and kernels.
