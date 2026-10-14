"""Same arbitrary one-direction activation rights; exact integer-budget prefix loss."""
from pathlib import Path
import ast,json,hashlib,time
import numpy as np,pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix
from threadpoolctl import threadpool_limits
threadpool_limits(1)
setup=Path('src/market/round5_plan.py').read_text(encoding='utf-8').split('rows=[];selected=[];traces=[];tic=')[0]
setup=setup.replace("O=Path('results/experiments/reservation_v5')","O=Path('results/experiments/reservation_v7/strong')")
exec(compile(setup,'frozen_predicted_information','exec'),globals())

def strong(p,sc):
    ncall=64;K=int(round(B0/(r0*dt)));assert abs(K*r0*dt-B0)<1e-9
    # Baseline c,d,half-hour modes; fully activated c,d,modes; epigraph h.
    ac,ad,am,h=240,304,368,432;cursor=496;prefix=[]
    for k in range(64):
        count=k+1
        if count<=K:prefix.append(None)
        else:prefix.append((cursor,cursor+1));cursor+=1+count
    y=cursor;cursor+=9;need=cursor;cursor+=len(sc)*64;nv=cursor
    lo=np.zeros(nv);hi=np.full(nv,np.inf);hi[:192]=P;hi[192:240]=1;hi[ac:am]=P;hi[am:h]=1;hi[h:496]=r0/eta;lo[y:y+9]=emin;hi[y:y+9]=emax;hi[need:]=r0
    integer=np.zeros(nv);integer[192:240]=1;integer[am:h]=1
    obj=np.zeros(nv);obj[:96]=dt*(p+20);obj[96:192]=dt*(20-p)
    rows=[];lbs=[];ubs=[]
    def add(row,l=-np.inf,u=np.inf):rows.append(row);lbs.append(l);ubs.append(u)
    def energy(t):return {**{i:eta*dt for i in range(t+1)},**{96+i:-dt/eta for i in range(t+1)}}
    def merge(a,b,scale=1):
        z=a.copy()
        for i,v in b.items():z[i]=z.get(i,0)+scale*v
        return z
    for t in range(96):
        add({t:1,192+t//2:-P},u=0);add({96+t:1,192+t//2:P},u=P);add(energy(t),emin-e0,emax-e0)
        if t%2==0:add({t:1,t+1:-1},0,0);add({96+t:1,97+t:-1},0,0)
    add(energy(95),0,0)
    loss_expr=[]
    for k,t in enumerate(call):
        t=int(t);add({ac+k:1,am+k:-P},u=0);add({ad+k:1,am+k:P},u=P)
        add({ad+k:1,ac+k:-1,96+t:-1,t:1},r0,r0)
        add({h+k:1,t:-eta,96+t:1/eta,ac+k:eta,ad+k:-1/eta},0,0)
        if prefix[k] is None:loss={h+j:dt for j in range(k+1)}
        else:
            zz,uu=prefix[k];loss={zz:K*dt}
            for j in range(k+1):add({uu+j:1,h+j:-1,zz:1},l=0);loss[uu+j]=dt
        loss_expr.append(loss);add(merge(energy(t),loss,-1),l=emin-e0)
    # y starts no higher than the true minimum end-of-window state.
    add(merge(merge({y:1},energy(87),-1),loss_expr[-1]),u=e0)
    envelope=np.max(sc,axis=0)
    for k,t in enumerate(rec):
        L=max(-P,(envelope[t]-cap)/2)
        if L>P:return None,{'status':2,'message':'recovery power infeasible'}
        add({y+k+1:1,y+k:-1},u=float(F(L)*dt))
        add(merge({y+k+1:1},energy(int(t)),-1),u=e0)
    add({y+8:1},e0,e0)
    for s in range(len(sc)):
        for t in outside:add({int(t):2,96+int(t):-2},u=cap-sc[s,t])
        for k,t in enumerate(call):add({int(t):1,96+int(t):-1,need+s*64+k:-1},u=(cap-sc[s,t])/2)
        add({need+s*64+k:dt for k in range(64)},u=B0)
    mat=lil_matrix((len(rows),nv))
    for k,row in enumerate(rows):
        for j,v in row.items():mat[k,j]=v
    sol=milp(obj,integrality=integer,bounds=Bounds(lo,hi),constraints=LinearConstraint(mat.tocsc(),lbs,ubs),options={'time_limit':60,'mip_rel_gap':1e-7})
    if not sol.success:return None,{'status':int(sol.status),'message':sol.message}
    q=sol.x[96:192]-sol.x[:96]
    hh=F(q[call])-F(q[call]+r0);eb=e0+np.cumsum(F(q)*dt)
    slack=min(eb[t]-dt*np.sort(hh[:k+1])[-min(K,k+1):].sum()-emin for k,t in enumerate(call))
    assert slack>=-1e-6
    return q,{'status':0,'gap':float(sol.mip_gap),'predicted_baseline_profit':float(-2*sol.fun),'min_robust_soc_slack':float(slack)}

if __name__=='__main__':
    rows=[];traces=[];tic=time.monotonic()
    for day in range(80):
        p=np.repeat(pred[day],2);sc=np.vstack([mean[(286+day)%7]+res[cfg['calibration_residual_ids']]+loss_margin,U])
        q,meta=strong(p,sc)
        if q is None:raise RuntimeError(str((day,meta)))
        rr=np.zeros(96);rr[call]=r0;assert all(service_pass(controller(q,s,rr,B0)[1]) for s in sc)
        rows.append(dict(day=day,method='strong',r_each_MW=r0,budget_each_MWh=B0,refund=0,accepted=False,selected_k=1,l_floor=0,**meta))
        for t in range(96):traces.append(dict(day=day,method='strong',slot=t,q_DA_each=q[t]))
        if (day+1)%10==0:print('strong planned',day+1,round(time.monotonic()-tic,1),flush=True)
    acdir=O/'AC';acdir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(acdir/'E2_pilot_daily.csv',index=False);pd.DataFrame(traces).to_csv(acdir/'E2_pilot_traces.csv',index=False);(acdir/'pilot_manifest.json').write_text(json.dumps(cfg,indent=2))
    (O/'frozen_plan_manifest.json').write_text(json.dumps({'actual_prices_read':False,'rights':'0<=a<=r0,sum(a*dt)<=B0; same as original reference','K':4,'files':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in acdir.glob('*.csv')}},indent=2))
    print('strong plans frozen',flush=True)
