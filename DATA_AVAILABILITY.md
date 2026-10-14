# Data availability

The public figure example contains exactly the two seven-day windows used by the paper's
Figs. 2 and 3 (5–18 January 2026), with 672 half-hour target observations and operating inputs,
plus the 1,024 preceding observations required by the first forecast. Subsequent forecasts
advance chronologically and reuse newly observed history. Only the target and seven model
covariates are disclosed. Comparator predictions are limited to those same 14 days.
The original LoRA adapter is included, with no training samples embedded as data files.

The figure inputs do not include saved LoRA predictions, full training/validation records,
complete 80-day evaluation series or raw monthly files. The earlier single-day data example
has been removed from this version to avoid releasing an additional unrelated window.

Eligible market participants may apply for complete market data through the Zhejiang Power
Exchange (https://www.zjpx.com.cn/), subject to registration, eligibility and authorization.
API interfaces and credentials must be obtained from the provider. This repository supplies
no unrestricted public API or access credentials.

SimBench-derived topology and profiles are supplied separately under their original data
terms. They are research simulations, not field measurements of Zhejiang's network.
The authors' complete private replication archive is not a public distribution artifact.
