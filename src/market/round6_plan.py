from pathlib import Path
import json,hashlib,time
prefix=Path('src/market/round5_plan.py').read_text(encoding='utf-8').split('rows=[];selected=[];traces=[];tic=')[0]
exec(compile(prefix,'audited_planning_setup','exec'),globals())
O=Path('results/experiments/reservation_v6');O.mkdir(exist_ok=True)
peak=float(U.max());protocol6={'gamma':[0,.5,1,1.5,2],'cap_formula':'max(U)-gamma*P_each','calibration_only':True,'same_80_days':True,'same_models_and_beta':True,'infeasible':'retained; no profit invented for absent B2','figures':'full80 totals only for wholly feasible regimes; infeasible boundary separate','highest_service_day_rule':'max test-day net demand, independent of prices/profits; illustrative, not typical','supersedes_quantile_caps':'quantile cases below necessary total-power bound; recorded before core sweep','predecessor':protocol}
(O/'protocol.json').write_text(json.dumps(protocol6,indent=2,ensure_ascii=False),encoding='utf-8')
allrows=[];allsel=[];tic=time.monotonic()
for gamma in protocol6['gamma']:
    case='g'+str(gamma).replace('.','p');C=O/case;C.mkdir(exist_ok=True);cap=peak-gamma*P;rows=[];selected=[];traces=[]
    for day in range(80):
        p=np.repeat(pred[day],2);sc=np.vstack([mean[(286+day)%7]+res[cfg['calibration_residual_ids']]+loss_margin,U]);rr=np.zeros(N);rr[call]=r0
        qb,mb=solve_flexible(p,sc,r0,e0)
        bs=[controller(qb,s,rr,B0) for s in sc] if qb is not None else []
        bp=bool(qb is not None and all(service_pass(v[1]) for v in bs));candidates=[]
        for k in [1,2,4,8]:
            result,meta=optimize(p,sc,'qualified',budget_cap=k*B0);G=L=score=np.nan;ok=False;q=None
            if result is not None:
                q=result['q'];cs=[controller(q,s,rr,k*B0) for s in sc];ok=all(service_pass(v[1]) for v in cs)
                if bp:
                    G=float(2*dt*np.sum(p*(q-qb)-20*(abs(q)-abs(qb))))
                    L=float(np.mean([settle(q,v[0],p,p)['activation_adjustment']-settle(qb,w[0],p,p)['activation_adjustment'] for v,w in zip(cs[:-1],bs[:-1])]))
                    score=G-max(0,L)
            eligible=bool(bp and ok and score>1e-6)
            row=dict(gamma=gamma,cap=cap,day=day,k=k,reference_status=mb['status'],reference_certificate=bp,status=meta['status'],gap=meta.get('gap',np.nan),candidate_certificate=ok,Ghat=G,Lhat=L,score=score,eligible=eligible)
            rows.append(row);candidates.append((row,q))
        good=[v for v in candidates if v[0]['eligible']];best=max([v[0]['score'] for v in good],default=0.)
        pick=min([v for v in good if v[0]['score']>=best-1e-6],key=lambda v:v[0]['k']) if good else None
        for method,use in [('B2',None),('menu',pick)]:
            if not bp:
                selected.append(dict(gamma=gamma,cap=cap,day=day,method=method,available=False,selected_k=np.nan,accepted=False,reference_status=mb['status']));continue
            q=qb if use is None else use[1];k=1 if use is None else use[0]['k'];G=0. if use is None else use[0]['Ghat'];L=0. if use is None else use[0]['Lhat']
            selected.append(dict(gamma=gamma,cap=cap,day=day,method=method,available=True,selected_k=k,accepted=use is not None,Ghat=G,Lhat=L,l_floor=max(0,L),forecast_refund=.5*(G+max(0,L)),refund=0.,r_each_MW=r0,budget_each_MWh=k*B0))
            for t in range(96):traces.append(dict(day=day,method=method,slot=t,q_DA_each=q[t]))
        if (day+1)%40==0:print(case,day+1,round(time.monotonic()-tic,1),flush=True)
    pd.DataFrame(rows).to_csv(C/'candidates.csv',index=False);pd.DataFrame(selected).to_csv(C/'E2_pilot_daily.csv',index=False);pd.DataFrame(traces).to_csv(C/'E2_pilot_traces.csv',index=False)
    cc=cfg.copy();cc['import_cap_MW']=cap;(C/'pilot_manifest.json').write_text(json.dumps(cc,indent=2));allrows+=rows;allsel+=selected
pd.DataFrame(allrows).to_csv(O/'all_candidates.csv',index=False);pd.DataFrame(allsel).to_csv(O/'all_selection.csv',index=False)
(O/'frozen_hashes.json').write_text(json.dumps({str(f.relative_to(O)):hashlib.sha256(f.read_bytes()).hexdigest() for f in O.glob('g*/*') if f.is_file()},indent=2))
print(pd.DataFrame(allsel).groupby(['gamma','method'])[['available','accepted']].sum().to_string(),flush=True)
