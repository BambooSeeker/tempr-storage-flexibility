# Final manuscript exhibit mapping

| Exhibit | Source / reconstruction |
|---|---|
| Fig. 1 | artwork/Fig1_framework.png; approved manual layout; scripts/repair_fig1_curves.py repairs native curve paths in the author's PPT using data/fig1_sources |
| Fig. 2 | scripts/plot_forecasts.py → Forecast_month |
| Fig. 3 | scripts/plot_forecasts.py → Forecast_week |
| Fig. 4 | scripts/plot_allocation.py → Forecast_ablation |
| Fig. 5 | scripts/plot_allocation.py → Forecast_operating_states |
| Fig. 6 | scripts/plot_service.py → Fig8_typical_import_days |
| Fig. 7 | scripts/plot_value.py → Fig1_economic_trajectory |
| Fig. 8 | scripts/plot_allocation.py → Fig7_typical_budget_responses |
| Fig. 9 | scripts/plot_value.py → Fig4_daily_mechanism |
| Table I | config/protocol.json; physical and service parameters |
| Table II | corrected forecast/outputs; official daily evaluators; ascending MAE |
| Table III | recorded timing evidence; exported manuscript entries, not a fresh timing benchmark |
| Table IV | reservation_v6 pressure_summary and settled_pressure_daily |
| Table V | reservation_v5/v8 complete AC ledgers and selections; recomputed settlement totals |
| Table VI | recorded day-79 mechanism/position evidence; exported manuscript entries |

`scripts/rebuild_tables.py` exports all six final rounded manuscript tables from
config/manuscript_tables.json and separately recomputes forecast, budget and settlement
values. Exporting a rounded table is not itself a fresh experiment. `verify_ledgers.py` checks
the archived main, pressure, alternative-reference and maximum-budget totals against their
original source tables. Figure aliases Fig01–Fig09 are written to generated/figures.

The legacy names T1–T9 under data/reference_tables are source-audit tables, not current manuscript
numbering. For selected days versus full 80 days, retain each table's stated denominator.
