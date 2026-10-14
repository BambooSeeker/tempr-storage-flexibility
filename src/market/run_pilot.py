"""Audited function definitions loaded by the planner into its parameter context.
Do not execute this module as a standalone experiment."""

def solve_flexible(p,sc,r,e0,final=False,idle=False):
    S=len(sc);nv=2*N+48+S*len(call);hoff=2*N+48
    obj=np.zeros(nv);obj[:N]=dt*(20+p);obj[N:2*N]=dt*(20-p)
    A=[];b=[];eq=[];rhs=[]
    for t in range(N):
        mode=2*N+t//2
        A.append({t:1,mode:-P});b.append(0)
        A.append({N+t:1,mode:P});b.append(P)
        if t in call:A.append({N+t:1,t:-1});b.append(P-r)
        energy={j:eta*dt for j in range(t+1)};energy.update({N+j:-dt/eta for j in range(t+1)})
        A.append(energy);b.append(emax-e0)
        C=min(H,max(0,(min(t,87)-23)*dt))*r if 24<=t<=87 else 0
        A.append({k:-v for k,v in energy.items()});b.append(e0-emin-C/eta)
    # Available downward net-injection headroom in recovery; scenario replay is additionally mandatory for PCC-aware certificate.
    A.append({**{int(t):dt for t in rec},**{int(N+t):-dt for t in rec}});b.append(len(rec)*P*dt-H*r/eta**2)
    for s in range(S):
        for t in outside:A.append({int(t):2.,int(N+t):-2.});b.append(cap-sc[s,t])
        for k,t in enumerate(call):A.append({int(t):2.,int(N+t):-2.,hoff+s*len(call)+k:-1.});b.append(cap-sc[s,t])
        A.append({hoff+s*len(call)+k:dt for k in range(len(call))});b.append(2*H*r)
    for t in range(0,N,2):
        eq.append({t:1,t+1:-1});rhs.append(0)
        eq.append({N+t:1,N+t+1:-1});rhs.append(0)
    if final or anchor:eq.append({**{t:eta*dt for t in range(N)},**{N+t:-dt/eta for t in range(N)}});rhs.append((e0 if anchor else .5*EM)-e0)
    mat=lil_matrix((len(A)+len(eq),nv))
    for k,row in enumerate(A+eq):
        for j,v in row.items():mat[k,j]=v
    lower=np.r_[np.full(len(A),-np.inf),rhs];upper=np.r_[b,rhs]
    ub=np.r_[np.full(2*N,P),np.ones(48),np.full(S*len(call),2*r)]
    if idle:ub[call]=0;ub[N+call]=0
    integ=np.zeros(nv);integ[2*N:2*N+48]=1
    out=milp(obj,integrality=integ,bounds=Bounds(np.zeros(nv),ub),constraints=LinearConstraint(mat.tocsc(),lower,upper),options={'time_limit':5,'mip_rel_gap':1e-6})
    if not out.success:return None,{'status':int(out.status),'message':out.message}
    return out.x[N:2*N]-out.x[:N],{'status':0,'predicted_baseline_profit':float(-out.fun*2),'solver_gap':float(out.mip_gap)}

def settle(q,x,pda,prt):
    R=2*dt*np.sum(pda*q+prt*(x-q));deg=2*20*dt*np.sum(np.abs(x));Rbase=2*dt*np.sum(pda*q);dbase=2*20*dt*np.sum(abs(q));S=-(R-Rbase)+(deg-dbase)
    return dict(energy_revenue=R,degradation=deg,baseline_profit=Rbase-dbase,activation_adjustment=S,owner_excl_F_refund=R-deg+S)

def service_pass(diag):
    return (diag['max_import_shortfall_MW']<1e-7 and diag['recovery_deficit_MWh']<1e-7 and diag['soc_min']>=.1-1e-7 and diag['soc_max']<=.9+1e-7 and diag['budget_violation']<1e-7)
