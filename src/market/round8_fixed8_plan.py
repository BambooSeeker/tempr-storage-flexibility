"""Matched maximum-budget comparison; frozen forecasts and unchanged selection rule."""
from pathlib import Path
import json,hashlib,time
import numpy as np,pandas as pd
setup=Path('src/market/round5_plan.py').read_text(encoding='utf-8').split('rows=[];selected=[];traces=[];tic=')[0]
setup=setup.replace("O=Path('results/experiments/reservation_v5')","O=Path('results/experiments/reservation_v8')")
exec(compile(setup,'predicted_information_only','exec'),globals())
oldc=pd.read_csv('results/experiments/reservation_v5/candidates.csv')
oldsel=pd.read_csv('results/experiments/reservation_v5/selection.csv')
oldtr=pd.read_csv('results/experiments/reservation_v5/planned_traces.csv')
rows=[];traces=[];comparisons=[];tic=time.monotonic()
for day in range(80):
    p=np.repeat(pred[day],2);sc=np.vstack([mean[(286+day)%7]+res[cfg['calibration_residual_ids']]+loss_margin,U])
    c=oldc[(oldc.day==day)&(oldc.k==8)].iloc[0]
    result,meta=optimize(p,sc,'qualified',budget_cap=8*B0);assert result is not None
    qb=oldtr[(oldtr.day==day)&(oldtr.method=='B2')].sort_values('slot').q_DA_each.to_numpy()
    q=result['q'];G=float(2*dt*np.sum(p*(q-qb)-20*(abs(q)-abs(qb))))
    assert abs(G-c.Ghat)<1e-5,'Different ex-ante optimum'
    accepted=bool(c.eligible);q=q if accepted else qb;k=8 if accepted else 1
    menu=oldsel[(oldsel.day==day)&(oldsel.method=='menu')].iloc[0]
    qm=oldtr[(oldtr.day==day)&(oldtr.method=='menu')].sort_values('slot').q_DA_each.to_numpy()
    comparisons.append(dict(day=day,fixed8_accepted=accepted,menu_k=int(menu.selected_k),menu_score=float(menu.Ghat-menu.l_floor),fixed8_score=float(c.score),max_plan_difference=float(np.max(abs(q-qm)))))
    rows.append(dict(day=day,method='fixed8',selected_k=k,accepted=accepted,Ghat=float(c.Ghat) if accepted else 0,Lhat=float(c.Lhat) if accepted else 0,l_floor=max(0,float(c.Lhat)) if accepted else 0,refund=0,r_each_MW=r0,budget_each_MWh=k*B0))
    for t in range(96):traces.append(dict(day=day,method='fixed8',slot=t,q_DA_each=q[t]))
ac=O/'AC';ac.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(ac/'E2_pilot_daily.csv',index=False);pd.DataFrame(traces).to_csv(ac/'E2_pilot_traces.csv',index=False);(ac/'pilot_manifest.json').write_text(json.dumps(cfg,indent=2))
pd.DataFrame(comparisons).to_csv(O/'fixed8_vs_menu_plans.csv',index=False)
(O/'protocol.json').write_text(json.dumps({'purpose':'fixed maximum budget against minimal near-optimal budget','fixed_k':8,'unchanged_forecast':True,'no_new_penalty':True,'prior_candidate_objectives_reproduced':True,'actual_prices_read':False,'replay':'if plans or binding budgets differ, matched 80-day AC'},indent=2))
(O/'frozen_plan_manifest.json').write_text(json.dumps({'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ac.glob('*.csv')},'forecast_sha256':protocol['forecast_sha256'],'actual_prices_read':False},indent=2))
print(pd.DataFrame(comparisons).agg({'max_plan_difference':['max'],'fixed8_accepted':['sum']}).to_string(),round(time.monotonic()-tic,1))
