# Model source inventory

All final-paper model implementations are public, including training and evaluation code.
Data availability and trained-artifact availability are separate from code availability.

| Paper model | Public implementation | Example execution |
|---|---|---|
| Chronos-2 + inputs + LoRA | run_chronos2_exog_lora.py | Original adapter included; live Fig. 2 / Fig. 3 inference |
| Chronos-2 + operating inputs | run_chronos2_four.py | Pretrained base downloadable; full original inference code |
| Chronos-2 price-only | run_chronos2_four.py | Pretrained base downloadable; full original inference code |
| Chronos-Bolt-small | run_chronos_zero.py | Pretrained model downloadable; original inference code |
| LightGBM | run_lightgbm.py | Full training and prediction source |
| SVR | run_svr_pricelags.py | Full training and prediction source |
| ARIMA | run_arima.py | Original rolling fitting and prediction source |
| LSTM residual | run_lstm_residual_delta.py | Full training and prediction source |
| Transformer residual | run_transformer_residual_delta.py | Full training and prediction source |
| Prophet | run_prophet.py | Original rolling fitting and prediction source |
| Previous day | shared_data.py | Baseline construction source |
| PatchTST residual | run_patchtst_residual_delta.py | Full training and prediction source |

Paths above are relative to forecast/code. Other original variants are retained in that
directory. Allocation, causal execution, AC evaluation and settlement code is in src/market.

The original LightGBM/SVR/neural comparator scripts saved prediction CSVs and metrics, without
serializing their final trained models. The inspected original experiment directory contains
the LoRA adapter but no corresponding fitted comparator artifacts. Therefore the lightweight
figure command computes LoRA live and overlays the published comparison results. The complete
training source remains available for researchers with authorized full training inputs.
Retraining would create new model artifacts; those must not be presented as the original
saved comparator checkpoint. This limitation does not restrict the public source code.
