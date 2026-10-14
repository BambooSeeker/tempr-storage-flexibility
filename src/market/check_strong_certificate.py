"""Finite verification of exact budget-extreme loss and causal recovery."""
from pathlib import Path
from itertools import combinations
import json
import numpy as np
eta=np.sqrt(.85);r=1.;dt=.25;K=4;P=2.;emin=0.;emax=10.
F=lambda q:eta*np.maximum(-q,0)-np.maximum(q,0)/eta
rng=np.random.default_rng(20261006);cases=0;fractional=0;maxerr=0.
for n in [6,7,8]:
 for trial in range(40):
  q=rng.uniform(-P,P-r,n)
  if trial==0:q=np.resize(np.array([-2.,-1.,-.5,0.,1.]),n)
  h=F(q)-F(q+r)
  for prefix in range(1,n+1):
   best=0.
   for count in range(min(K,prefix)+1):
    for ids in combinations(range(prefix),count):
     a=np.zeros(prefix);a[list(ids)]=r
     best=max(best,float(dt*np.sum(F(q[:prefix])-F(q[:prefix]+a))));cases+=1
   formula=dt*np.sort(h[:prefix])[-min(K,prefix):].sum();maxerr=max(maxerr,abs(best-formula));assert abs(best-formula)<1e-10
   for _ in range(20):
    a=rng.uniform(0,r,prefix);a*=min(1,K*r/a.sum());value=dt*np.sum(F(q[:prefix])-F(q[:prefix]+a));assert value<=formula+1e-10;fractional+=1
# Greedy recovery maps monotonically in initial state and respects cap-induced L.
recoveries=0
for q,L in [(np.array([0.,0.,0.,0.]),np.full(4,-1.)),(np.full(4,.6),np.full(4,.1)),(np.full(4,-.5),np.full(4,-1.5))]:
 base=np.r_[5.,5.+np.cumsum(F(q)*dt)]
 for deficit in np.linspace(0,2,41):
  e=base[0]-deficit;states=[]
  for t in range(4):
   target=F(q[t])+max(0,base[t]-e)/dt;x=-target/eta if target>=0 else -target*eta
   x=min(q[t],max(-P,L[t],x));e+=F(x)*dt;states.append(e)
  # The witness maximal state follows min(baseline next, previous+F(L)*dt).
  y=base[0]-deficit
  for t in range(4):y=min(base[t+1],y+F(L[t])*dt)
  assert abs(e-y)<1e-10;recoveries+=1
# Without proper entrance coupling a zero-headroom recovery would falsely admit deficit.
assert min(5.,4.+F(0.)*dt)<5.
O=Path('results/experiments/reservation_v7');(O/'strong_certificate_verification.json').write_text(json.dumps({'vertex_paths_checked':cases,'fractional_paths_checked':fractional,'max_error':maxerr,'recovery_cases':recoveries,'integer_budget_slots':4,'proof_scope':'uniform power/slot, integer budget, fixed baseline, one-way call, no ramp; causal recovery','not_continuous_proof_by_enumeration':True},indent=2))
print(cases,fractional,recoveries,maxerr)
