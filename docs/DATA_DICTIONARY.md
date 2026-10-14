# Input schema and units

Figure example CSVs use local market timestamps, with 30-minute steps.
`data/figures/initial_history.csv`: time, price_day_ahead and seven covariates, 1024 rows.
`operating_inputs.csv`: time and seven covariates, 672 rows; no target column.
`observations.csv`: time and price_day_ahead, revealed day by day after prediction.
`published_comparators.csv`: time and published baseline predictions, only for final plotting.
No saved LoRA predictions are bundled in the input data.

| Feature | Original column expression | Unit |
|---|---|---|
| demand | load_day_ahead_pred − elec_exter_plan | MW |
| supply | elec_gene_total_pred | MW |
| bidding_space | load_day_ahead_pred − elec_exter_plan − elec_fix_out_plan − energy_hydro_renewable − photo_gene_total_pred − wind_gene_total_pred | MW |
| coal_gas_nuclear_pred | elec_gene_total_pred − photo_gene_total_pred − wind_gene_total_pred − water_gene_total_pred | MW |
| renewable_penetration | (wind_gene_total_pred + photo_gene_total_pred)/(supply + 1e-6) | ratio |
| photo_gene_total_pred | original PV forecast | MW |
| wind_gene_total_pred | original wind forecast | MW |

Full monthly input schema is recorded in config/raw_input_schema.json; authorized users can
map their data to it. No credentials or download endpoint are encoded. The shared_data.py
loader filters the study node, orders and deduplicates timestamps and handles missing values
using the original preprocessing logic.

Network profiles have 366 × 96 quarter-hour rows. net is total net demand; load, qload and
sgen hold bus/device profiles. The first 260 days form historical information; evaluation uses
day indices 286–365. Configured battery buses are 124 and 132. Two identical batteries are
scheduled symmetrically; budget_each_MWh and q_DA_each refer to one battery. Quarter-hour
traces and cumulative activation totals use dt=0.25 hours.
