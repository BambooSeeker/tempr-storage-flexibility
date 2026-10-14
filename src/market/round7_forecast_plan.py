"""E04 planner: never reads actual DA/RT files. Predicted prices only.
Extract prior audited function definitions without executing module imports.
"""
from pathlib import Path
import ast,json,hashlib,time
import numpy as np,pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix
from threadpoolctl import threadpool_limits
threadpool_limits(1)
O=Path('results/experiments/reservation_v7/unfinetuned');O.mkdir(exist_ok=True)
protocol=json.loads(Path('config/protocol.json').read_text(encoding='utf-8'))
cfg=protocol['reference'];P=cfg['P_each_MW'];EM=cfg['E_each_MWh'];eta=np.sqrt(.85);dt=.25;N=96;emin=.1*EM;emax=.9*EM;e0=.5*EM;r0=cfg['initial_r_each_MW'];B0=r0;H=1.;anchor=True;cap=cfg['import_cap_MW'];loss_margin=cfg['loss_margin_MW'];call=np.arange(24,88);rec=np.arange(88,96);outside=np.r_[np.arange(24),rec]
net=np.load('data/network/network_profiles.npz')['net'].reshape(366,96)
hist=net[:260].copy();del net
mean=np.array([hist[:210][np.arange(210)%7==i].mean(0) for i in range(7)]);res=np.array([hist[j]-mean[j%7] for j in range(210,260)]);U=hist.max(0)+loss_margin
pf=Path(r'forecast/outputs/daily_pred_Chronos2_ExogDirect.csv');protocol['forecast_sha256']=hashlib.sha256(pf.read_bytes()).hexdigest();pred=pd.read_csv(pf).iloc[:,1:].to_numpy()
for name,fs in [('src/market/round2.py',['F','controller','optimize']),('src/market/run_pilot.py',['solve_flexible','settle','service_pass'])]:
    tree=ast.parse(Path(name).read_text(encoding='utf-8'));nodes=[a for a in tree.body if isinstance(a,ast.FunctionDef) and a.name in fs];exec(compile(ast.Module(body=nodes,type_ignores=[]),name,'exec'),globals())
rows=[];selected=[];traces=[];tic=time.monotonic()
for day in range(80):
    p=np.repeat(pred[day],2);sc=np.vstack([mean[(286+day)%7]+res[cfg['calibration_residual_ids']]+loss_margin,U]);rr=np.zeros(N);rr[call]=r0
    qb,mb=solve_flexible(p,sc,r0,e0);assert qb is not None
    bs=[controller(qb,s,rr,B0) for s in sc];bp=all(service_pass(v[1]) for v in bs);candidates=[]
    for k in protocol['budget_multiples']:
        result,meta=optimize(p,sc,'qualified',budget_cap=k*B0)
        G=L=score=np.nan;ok=False;q=None
        if result is not None:
            q=result['q'];cs=[controller(q,s,rr,k*B0) for s in sc];ok=bp and all(service_pass(v[1]) for v in cs)
            G=float(2*dt*np.sum(p*(q-qb)-20*(abs(q)-abs(qb))))
            L=float(np.mean([settle(q,v[0],p,p)['activation_adjustment']-settle(qb,w[0],p,p)['activation_adjustment'] for v,w in zip(cs[:-1],bs[:-1])]))
            score=G-max(0,L)
        eligible=bool(ok and score>protocol['positive_threshold'])
        row=dict(day=day,k=k,status=meta['status'],gap=meta.get('gap',np.nan),certificate_pass=ok,Ghat=G,Lhat=L,score=score,eligible=eligible)
        rows.append(row);candidates.append((row,q))
    good=[v for v in candidates if v[0]['eligible']]
    best=max([v[0]['score'] for v in good],default=0.)
    menu=min([v for v in good if v[0]['score']>=best-protocol['tie_tolerance']],key=lambda v:v[0]['k']) if good else None
    fixed=next(v for v in candidates if v[0]['k']==2);fixed=fixed if fixed[0]['eligible'] else None
    for method,pick in [('B2',None),('menu',menu),('fixed2',fixed)]:
        q=qb if pick is None else pick[1];k=1 if pick is None else pick[0]['k'];G=0. if pick is None else pick[0]['Ghat'];L=0. if pick is None else pick[0]['Lhat']
        selected.append(dict(day=day,method=method,selected_k=k,accepted=pick is not None,Ghat=G,Lhat=L,l_floor=max(0,L),forecast_refund=.5*(G+max(0,L)),refund=0.,r_each_MW=r0,budget_each_MWh=k*B0))
        for t in range(96):traces.append(dict(day=day,method=method,slot=t,q_DA_each=q[t]))
    if (day+1)%20==0:print('planned',day+1,round(time.monotonic()-tic,1),flush=True)
pd.DataFrame(rows).to_csv(O/'candidates.csv',index=False);pd.DataFrame(selected).to_csv(O/'selection.csv',index=False);pd.DataFrame(traces).to_csv(O/'planned_traces.csv',index=False)
ac=O/'AC';ac.mkdir(exist_ok=True);pd.DataFrame(selected).to_csv(ac/'E2_pilot_daily.csv',index=False);pd.DataFrame(traces).to_csv(ac/'E2_pilot_traces.csv',index=False);(ac/'pilot_manifest.json').write_text(json.dumps(cfg,indent=2))
lock={'protocol':protocol,'planner_reads_actual_prices':False,'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [O/'candidates.csv',O/'selection.csv',O/'planned_traces.csv']}}
(O/'frozen_plan_manifest.json').write_text(json.dumps(lock,indent=2,ensure_ascii=False),encoding='utf-8');print('Plans frozen before realized-price accounting')
