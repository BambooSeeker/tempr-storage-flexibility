from pathlib import Path
import json,numpy as np,pandas as pd
O=Path('results/experiments/reservation_v7/strong/AC');B=Path(r'forecast');prof=np.load('data/network/network_profiles.npz');z=prof['net'].reshape(366,96);points=pd.read_csv(O/'E2_online_AC_15min.csv');rows=pd.read_csv(O/'E2_online_AC_daily.csv');cfg=json.loads((O/'pilot_manifest.json').read_text(encoding='utf-8'));pred=pd.read_csv(B/'outputs/daily_pred_Chronos2_ExogDirect_LoRA600.csv');dates=pd.to_datetime(pred.iloc[:,0]);y=pd.read_csv(B/'outputs/y_test.csv').set_index('forecast_start_time').reindex(pred.iloc[:,0]);DA=np.repeat(y.to_numpy(),2,axis=1)
frames=[]
for f in (B/'data').glob('*.csv'):
    d=pd.read_csv(f,usecols=['time','code','price_real']);frames.append(d[d.code.eq('YQFDC-01')])
d=pd.concat(frames);d.time=pd.to_datetime(d.time);RT=np.repeat(d.set_index('time').reindex([t+pd.Timedelta(minutes=30*k) for t in dates for k in range(48)]).price_real.to_numpy().reshape(80,48),2,axis=1)
mean=np.array([z[:210][np.arange(210)%7==i].mean(0) for i in range(7)]);dt=.25
for ri,row in rows.iterrows():
    day=int(row.day);m=row.method;a=points[(points.day==day)&points.method.eq(m)].sort_values('slot');q=a.q_DA_each.to_numpy();grossda=mean[(286+day)%7]+cfg['loss_margin_MW'];pccrt=a.pcc_mw.to_numpy();grossrt=pccrt+2*a.q_exec_each.to_numpy();pccda=grossda-2*q;R=row.energy_revenue
    cg=dt*np.sum(DA[day]*grossda+RT[day]*(grossrt-grossda));cn=dt*np.sum(DA[day]*pccda+RT[day]*(pccrt-pccda));err=abs(cg-R-cn);assert err<1e-6
    rows.loc[ri,'gross_energy_cost']=cg;rows.loc[ri,'net_external_energy_cost']=cn;rows.loc[ri,'grid_total_cost_excl_F']=cg+row.grid_contract_cost_excl_F;rows.loc[ri,'combined_external_cost']=cn+row.degradation;rows.loc[ri,'cash_identity_error']=err
rows.to_csv(O/'E2_online_AC_complete_ledger.csv',index=False)
summary=rows.groupby('method')[['net_owner_excl_F','grid_total_cost_excl_F','combined_external_cost']].sum().to_dict();summary['max_cash_identity_error']=float(rows.cash_identity_error.max());summary['scope']='actual AC losses and DA/RT settlement; gross network account includes fixed non-storage generation as net demand; common F omitted, internal transfers cancel'
(O/'E2_online_AC_complete_ledger_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))
