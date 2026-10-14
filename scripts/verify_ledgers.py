"""Rebuild core totals from frozen daily ledgers without model training."""
from pathlib import Path
import hashlib,json
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1]
checks={}
(R/'generated').mkdir(exist_ok=True)
manifest=pd.read_csv(R/'manifest.csv')
for a in manifest.itertuples():
 p=R/a.path
 assert p.exists(),a.path
 assert hashlib.sha256(p.read_bytes()).hexdigest()==a.sha256,a.path
checks['hashed_files']=len(manifest)
v=pd.read_csv(R/'results/experiments/reservation_v5/settled_daily.csv');a=v.groupby('method')[['owner_gain','buyer_saving','total_surplus']].sum();b=pd.read_csv(R/'data/reference_tables/T5_settlement.csv').set_index('method');assert np.allclose(a.sort_index(),b.sort_index(),atol=1e-7)
checks['settlement_table']=True
pred=pd.DataFrame(json.loads((R/'forecast/outputs/metrics_official_daily.json').read_text(encoding='utf-8')));t=pd.read_csv(R/'data/reference_tables/T1_forecast_metrics.csv');assert np.allclose(pred.mae,t.mae);assert np.allclose(pred.rmse,t.rmse)
records=[]
for model,case,selpath in [('LoRA','reservation_v5','selection.csv'),('Unfinetuned','reservation_v7/unfinetuned','selection.csv')]:
 base=R/'results/experiments'/case;d=pd.read_csv(base/'AC/E2_online_AC_complete_ledger.csv');sel=pd.read_csv(base/selpath);ref=d[d.method=='B2'].set_index('day')
 for method in ['B2','fixed2','menu']:
  x=d[d.method==method].set_index('day').sort_index();c=sel[sel.method==method].set_index('day').sort_index();g=x.net_owner_excl_F-ref.net_owner_excl_F;pay=np.where(c.accepted,.5*(g+c.l_floor),0)
  records.append(dict(predictor=model,method=method,owner_increment=(g-pay).sum(),grid_increment=(ref.grid_total_cost_excl_F-x.grid_total_cost_excl_F+pay).sum(),joint_increment=(ref.combined_external_cost-x.combined_external_cost).sum(),combined_external_cost=x.combined_external_cost.sum(),full_service_days=int(x.full_service_pass.sum())))
calc=pd.DataFrame(records).set_index(['predictor','method']).sort_index();target=pd.read_csv(R/'data/reference_tables/T2_prediction_and_allocation.csv').set_index(['predictor','method']).sort_index();assert np.allclose(calc,target[calc.columns],atol=1e-7,rtol=0)
assert np.allclose(calc.owner_increment+calc.grid_increment,calc.joint_increment,atol=1e-6)
checks['forecast_allocation_table']=True
raw=pd.read_csv(R/'results/experiments/reservation_v6/settled_pressure_daily.csv');table=pd.read_csv(R/'data/reference_tables/T4_service_requirements_main.csv')
for x in table.itertuples():
 d=raw[(raw.gamma==x.gamma)&(raw.method=='menu')];assert len(d)==80;assert int(d.full_service_pass.sum())==x.menu_pass
 if x.comparable_full_service:assert abs(d.joint_gain.sum()-x.joint_gain)<1e-6
 else:assert pd.isna(x.joint_gain)
checks['pressure_table']=True
base=pd.read_csv(R/'results/experiments/reservation_v5/AC/E2_online_AC_complete_ledger.csv');strong=pd.read_csv(R/'results/experiments/reservation_v7/strong/AC/E2_online_AC_complete_ledger.csv');expected=pd.read_csv(R/'data/reference_tables/T3_same_rights_reference.csv');br=base[base.method=='B2'].combined_external_cost.sum()
for method,d in [('Original rights',base[base.method=='B2']),('Exact-loss rights',strong),('Network-conditioned menu',base[base.method=='menu'])]:
 row=expected[expected.method==method].iloc[0];assert abs(br-d.combined_external_cost.sum()-row.joint_increment_vs_original)<1e-6
checks['strong_reference_table']=True

# Matched maximum budget: recalculate transfers and budget sums from raw records.
v8=R/'results/experiments/reservation_v8'
xx=pd.read_csv(v8/'AC/E2_online_AC_complete_ledger.csv').set_index('day').sort_index()
ss=pd.read_csv(v8/'AC/E2_pilot_daily.csv').set_index('day').sort_index()
rr=base[base.method=='B2'].set_index('day').sort_index()
gg=xx.net_owner_excl_F-rr.net_owner_excl_F
tt=np.where(ss.accepted,.5*(gg+ss.l_floor),0)
tab=pd.read_csv(R/'data/reference_tables/T7_matched_maximum_budget.csv').set_index('method')
assert abs((gg-tt).sum()-tab.loc['fixed8','owner_increment'])<1e-6
assert abs((rr.grid_total_cost_excl_F-xx.grid_total_cost_excl_F+tt).sum()-tab.loc['fixed8','grid_increment'])<1e-6
assert abs((rr.combined_external_cost-xx.combined_external_cost).sum()-tab.loc['fixed8','joint_increment'])<1e-6
assert xx.full_service_pass.all() and len(xx)==80
assert abs(2*ss.loc[ss.accepted,'budget_each_MWh'].sum()-tab.loc['fixed8','accepted_budget_sum_two_units_MWh'])<1e-7
saved=pd.read_csv(v8/'settled_daily.csv');assert np.allclose(saved.total_surplus,rr.combined_external_cost.to_numpy()-xx.combined_external_cost.to_numpy(),atol=1e-7)
pd8=pd.read_csv(R/'data/reference_tables/T8_period_decomposition.csv').set_index('period')
assert abs(pd8.loc['All 80 days','menu_minus_fixed8']-(tab.loc['menu','joint_increment']-tab.loc['fixed8','joint_increment']))<1e-6
assert abs(pd8.loc['Day 79','menu_joint']+pd8.loc['Other 79 days','menu_joint']-pd8.loc['All 80 days','menu_joint'])<1e-6
checks['matched_fixed8_and_period_tables']=True

(R/'generated/ledger_checks.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
