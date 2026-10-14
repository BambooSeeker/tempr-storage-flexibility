# Changelog

## 1.2.0, 2026-10-15

* Purge forecast target windows crossing train/validation/test boundaries; fit target scaling on training labels.
* Retrain the five affected comparators on the unchanged test period and refresh the two public comparison windows.
* Add regression checks for target disjointness, irregular time gaps and scaler isolation.
* Retain the original LoRA adapter, its frozen market forecasts and all economic experiment records.
* Update provenance, citation metadata and release validation.

## 1.1.0

Original adapter-based inference for 14 figure days, with 1,024 initial historical points.
