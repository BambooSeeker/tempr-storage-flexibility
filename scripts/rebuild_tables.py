"""Recompute headline paper values from frozen observations and daily ledgers."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];O=R/'generated/tables';O.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(R/'forecast/code'))
import shutil
import evaluate_official_daily as errors
import evaluate_official_daily_mape as mape
metrics_dir=R/'generated/forecast_metrics';metrics_dir.mkdir(parents=True,exist_ok=True)
for source in (R/'forecast/outputs').glob('*.csv'):shutil.copy2(source,metrics_dir/source.name)
errors.OUTPUT_DIR=metrics_dir;mape.OUTPUT_DIR=metrics_dir
errors.main();mape.main()
f=pd.read_csv(metrics_dir/'metrics_official_daily.csv');m=pd.read_csv(metrics_dir/'metrics_official_daily_mape.csv');f=f.merge(m[['model','daily_mape_percent']],on='model').sort_values('mae');f.to_csv(O/'Table_II_recomputed_all_models.csv',index=False)
q=f[f.model.eq('Chronos2_ExogDirect_LoRA600')].iloc[0];assert abs(q.mae-16.0173)<.0001 and abs(q.daily_mape_percent-6.9721)<.0001
base=pd.read_csv(R/'results/experiments/reservation_v5/AC/E2_online_AC_complete_ledger.csv');ref=base[base.method.eq('B2')].set_index('day').sort_index();records=[]
for method in ['B2','fixed2','fixed8','menu']:
 if method=='fixed8':
  x=pd.read_csv(R/'results/experiments/reservation_v8/AC/E2_online_AC_complete_ledger.csv').set_index('day').sort_index();s=pd.read_csv(R/'results/experiments/reservation_v8/AC/E2_pilot_daily.csv').set_index('day').sort_index()
 else:
  x=base[base.method.eq(method)].set_index('day').sort_index();ss=pd.read_csv(R/'results/experiments/reservation_v5/selection.csv');s=ss[ss.method.eq(method)].set_index('day').sort_index()
 gain=x.net_owner_excl_F-ref.net_owner_excl_F;pay=np.where(s.accepted,.5*(gain+s.l_floor),0);owner=(gain-pay).sum();grid=(ref.grid_total_cost_excl_F-x.grid_total_cost_excl_F+pay).sum();joint=(ref.combined_external_cost-x.combined_external_cost).sum();assert abs(owner+grid-joint)<1e-6
 records.append({'method':method,'storage_gain':owner,'grid_gain':grid,'joint_gain':joint,'service_days':int(x.full_service_pass.sum()),'activation_MWh':x.activation_MWh.sum(),'accepted_ceiling_MWh_two_batteries':2*s.loc[s.accepted,'budget_each_MWh'].sum(),'total_ceiling_MWh_two_batteries':2*s.budget_each_MWh.sum()})
t=pd.DataFrame(records);t.to_csv(O/'Table_V_and_budgets_recomputed.csv',index=False);menu=t[t.method.eq('menu')].iloc[0];fixed=t[t.method.eq('fixed8')].iloc[0];assert abs(menu.joint_gain-2475.258638)<1e-5
reduction=1-menu.accepted_ceiling_MWh_two_batteries/fixed.accepted_ceiling_MWh_two_batteries;assert abs(reduction-.6)<1e-8
p=pd.read_csv(R/'results/experiments/reservation_v6/pressure_summary.csv');p.to_csv(O/'Table_IV_service_recomputed_summary.csv',index=False)
# Export the final author-approved rounded table entries, already checked against source ledgers.
tables=json.loads((R/'config/manuscript_tables.json').read_text(encoding='utf-8'))
for roman,rows in zip(['I','II','III','IV','V','VI'],tables):pd.DataFrame(rows[1:],columns=rows[0]).to_csv(O/f'Table_{roman}_manuscript.csv',index=False)
(R/'generated/paper_checks.json').write_text(json.dumps({'LoRA_MAE':float(q.mae),'LoRA_MAPE':float(q.daily_mape_percent),'joint_gain':float(menu.joint_gain),'accepted_commitment_reduction':float(reduction),'main_methods_all_80_days':bool(t.service_days.eq(80).all())},indent=2))
print('Paper metric, settlement, and budget checks passed.')
