"""Independent AC checks for new pressure cases; reuse gamma1 only after plan match."""
from pathlib import Path
import subprocess,sys,os,json,shutil
from concurrent.futures import ThreadPoolExecutor
import pandas as pd,numpy as np
O=Path('results/experiments/reservation_v6');base=Path('results/experiments/reservation_v5/AC')
a=pd.read_csv(O/'g1/E2_pilot_traces.csv');b=pd.read_csv('results/experiments/reservation_v5/planned_traces.csv');b=b[b.method.isin(['B2','menu'])]
for m in ['B2','menu']:assert np.allclose(a[a.method==m].q_DA_each,b[b.method==m].q_DA_each,atol=1e-7,rtol=0)
for name in ['E2_online_AC_daily.csv','E2_online_AC_15min.csv','E2_online_AC_complete_ledger.csv']:
    d=pd.read_csv(base/name);d=d[d.method.isin(['B2','menu'])];d.to_csv(O/'g1'/name,index=False)
acsrc=Path('src/market/run_round5_ac.py').read_text(encoding='utf-8');ledsrc=Path('src/market/close_round5_ledger.py').read_text(encoding='utf-8')
jobs=[]
for case in ['g0','g0p5','g1p5']:
    d=pd.read_csv(O/case/'E2_pilot_daily.csv');assert len(d)==160 and d.available.all(), 'Keep incomplete cases out of full80 profit comparison'
    ap=Path('src/market')/('ac6_'+case+'.py');lp=Path('src/market')/('ledger6_'+case+'.py')
    ap.write_text(acsrc.replace('reservation_v5/AC','reservation_v6/'+case).replace("['menu','fixed2']","['B2','menu']"),encoding='utf-8')
    lp.write_text(ledsrc.replace('reservation_v5/AC','reservation_v6/'+case),encoding='utf-8');jobs.append((case,ap,lp))
def run(job):
    case,ap,lp=job
    with (O/case/'run.log').open('w',encoding='utf-8') as f:
        for script in [ap,lp]:
            r=subprocess.run([sys.executable,str(script)],stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            if r.returncode:raise RuntimeError(case+' failed; see run.log')
    print(case,'AC and ledger complete',flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,jobs))
print('all new AC cases complete',flush=True)
