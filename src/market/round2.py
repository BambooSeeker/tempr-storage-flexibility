"""Audited function definitions loaded by the planner into its parameter context.
Do not execute this module as a standalone experiment."""

def F(x):return eta*np.maximum(-x,0)-np.maximum(x,0)/eta

def controller(q,gross,rights=None,budget=None):
    eb=np.r_[e0,e0+np.cumsum(F(q)*dt)];e=e0;used=0.;out=[];soc=[e];sh=[];acts=[]
    for t in range(N):
        x=q[t];a=0
        if t in call:
            need=max(0,(gross[t]-cap)/2-q[t]);limit=P-q[t]
            if rights is not None:limit=min(limit,rights[t],max(0,budget-used)/dt)
            limit=min(limit,max(0,(e-emin)*eta/dt-q[t]))
            a=min(need,max(0,limit));x=q[t]+a;used+=a*dt
        elif t in rec:
            target=F(q[t])+max(0,eb[t]-e)/dt
            xt=-target/eta if target>=0 else -target*eta
            x=max(-P,min(q[t],xt),(gross[t]-cap)/2)
            # If baseline itself cannot meet outside-window import cap, record failure; do not issue a new call.
            x=min(q[t],x)
        e+=F(x)*dt;out.append(x);soc.append(e);sh.append(max(0,gross[t]-2*x-cap));acts.append(a)
    diag={'max_import_shortfall_MW':max(sh),'recovery_deficit_MWh':2*max(0,eb[-1]-e),'soc_min':min(soc)/EM,'soc_max':max(soc)/EM,'budget_violation':max(0,used-budget) if budget is not None else 0,'activation_MWh':2*used}
    return np.array(out),diag,np.array(acts)

def optimize(p,sc,kind,budget_cap=None):
    # Baseline c,d (192), baseline binary modes (48).
    if kind=='profile':nv=240+64+len(sc)*64;roff=240;hoff=304
    elif kind in ['causal','qualified','keep_power','keep_energy']:nv=240+64*3+9;wc=240;wd=304;wm=368;yy=432
    else:nv=240
    obj=np.zeros(nv);obj[:N]=dt*(p+20);obj[N:2*N]=dt*(20-p)
    lo=np.zeros(nv);up=np.full(nv,np.inf);up[:2*N]=P;up[192:240]=1;integ=np.zeros(nv);integ[192:240]=1
    rows=[];lbs=[];ubs=[]
    def add(row,l=-np.inf,u=np.inf):rows.append(row);lbs.append(l);ubs.append(u)
    def energy(t):return {**{i:eta*dt for i in range(t+1)},**{N+i:-dt/eta for i in range(t+1)}}
    def energy_add(a,b,scale=1):
        out=a.copy()
        for k,v in b.items():out[k]=out.get(k,0)+scale*v
        return out
    for t in range(N):
        add({t:1,192+t//2:-P},u=0);add({N+t:1,192+t//2:P},u=P);add(energy(t),emin-e0,emax-e0)
        if t%2==0:add({t:1,t+1:-1},0,0);add({N+t:1,N+t+1:-1},0,0)
    add(energy(95),0,0)
    if kind=='profile':
        up[roff:hoff]=r0;up[hoff:]=2*r0
        for k,t in enumerate(call):
            add({N+t:1,t:-1,roff+k:1},u=P)
            C=min(H,(t-23)*dt)*r0
            add(energy(t),l=emin-e0+C/eta)
            if k%2==0:add({roff+k:1,roff+k+1:-1},0,0)
        add({**{int(t):dt for t in rec},**{int(N+t):-dt for t in rec}},u=len(rec)*P*dt-B0/eta**2)
        for s in range(len(sc)):
            for t in outside:add({int(t):2,int(N+t):-2},u=cap-sc[s,t])
            for k,t in enumerate(call):
                h=hoff+s*64+k;add({int(t):2,int(N+t):-2,h:-1},u=cap-sc[s,t]);add({h:1,roff+k:-2},u=0)
            add({hoff+s*64+k:dt for k in range(64)},u=2*B0)
    elif kind in ['causal','qualified','keep_power','keep_energy']:
        up[wc:wm]=P;up[wm:yy]=1;integ[wm:yy]=1;lo[yy:]=emin;up[yy:]=emax
        low=(np.max(sc,axis=0)-cap)/2
        for t in outside:add({N+int(t):1,int(t):-1},l=low[t])
        for k,t in enumerate(call):
            add({wc+k:1,wm+k:-P},u=0);add({wd+k:1,wm+k:P},u=P)
            add({wd+k:1,wc+k:-1,N+t:-1,t:1},l=0);add({wd+k:1,wc+k:-1},l=low[t])
            if kind in ['qualified','keep_power']:add({wd+k:1,wc+k:-1,N+t:-1,t:1},u=r0)
            ew=energy(23);ew.update({wc+i:eta*dt for i in range(k+1)});ew.update({wd+i:-dt/eta for i in range(k+1)})
            add(ew,l=emin-e0)
        add(energy_add({yy:1},ew,-1),e0,e0)
        if kind in ['qualified','keep_energy']:add({**{wd+k:dt for k in range(64)},**{wc+k:-dt for k in range(64)},**{int(N+t):-dt for t in call},**{int(t):dt for t in call}},u=B0 if budget_cap is None else budget_cap)
        for k,t in enumerate(rec):
            L=max(-P,low[t])
            if L>P:return None,{'status':2,'message':'network bound above power rating'}
            add({yy+k+1:1,yy+k:-1},u=float(F(L)*dt))
            add(energy_add({yy+k+1:1},energy(int(t)),-1),u=e0)
        add(energy_add({yy+8:1},energy(95),-1),e0,e0)
    elif kind=='static':
        for t in range(N):add({N+t:1,t:-1},l=(np.max(sc[:,t])-cap)/2)
    elif kind!='market':raise ValueError(kind)
    mat=lil_matrix((len(rows),nv))
    for k,row in enumerate(rows):
        for j,v in row.items():mat[k,j]=v
    sol=milp(obj,integrality=integ,bounds=Bounds(lo,up),constraints=LinearConstraint(mat.tocsc(),lbs,ubs),options={'time_limit':10,'mip_rel_gap':1e-7})
    if not sol.success:return None,{'status':int(sol.status),'message':sol.message}
    q=sol.x[N:2*N]-sol.x[:N];rights=None
    if kind=='profile':rights=np.zeros(N);rights[call]=sol.x[roff:hoff]
    return {'q':q,'rights':rights}, {'status':0,'gap':float(sol.mip_gap),'predicted_baseline_profit':float(-2*sol.fun)}
