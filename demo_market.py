"""Re-solve the first released figure-day budget menu using released inputs only."""
from pathlib import Path
import os,json
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent
def main():
 os.chdir(R)
 if not (R/'generated/live_figures/live_predictions.csv').exists():raise SystemExit('Run python infer_figures.py first; this market example consumes its freshly computed LoRA predictions.')
 s=(R/'src/market/round5_plan.py').read_text(encoding='utf-8').split('rows=[];selected=[];traces=[];tic=')[0]
 s=s.replace("O=Path('results/experiments/reservation_v5');O.mkdir(exist_ok=True)","O=Path('generated/market_demo');O.mkdir(parents=True,exist_ok=True)")
 a=s.index('pf=Path(');b=s.index('\nfor name,fs',a)
 s=s[:a]+"pred=pd.read_csv('generated/live_figures/live_predictions.csv').LoRA_live.to_numpy()[:48].reshape(1,48)"+s[b:]
 g={};exec(compile(s,'released_demo_setup','exec'),g)
 p=np.repeat(g['pred'][0],2);sc=np.vstack([g['mean'][(286+42)%7]+g['res'][g['cfg']['calibration_residual_ids']]+g['loss_margin'],g['U']])
 qb,_=g['solve_flexible'](p,sc,g['r0'],g['e0']);assert qb is not None
 rr=np.zeros(96);rr[g['call']]=g['r0'];bs=[g['controller'](qb,x,rr,g['B0']) for x in sc]
 bp=all(g['service_pass'](x[1]) for x in bs);rows=[];schedules={'slot':np.arange(96),'reference_MW_each':qb}
 for k in g['protocol']['budget_multiples']:
  r,meta=g['optimize'](p,sc,'qualified',budget_cap=k*g['B0']);G=L=score=None;ok=False
  if r is not None:
   q=r['q'];cs=[g['controller'](q,x,rr,k*g['B0']) for x in sc]
   ok=bool(bp and all(g['service_pass'](x[1]) for x in cs))
   G=float(2*g['dt']*np.sum(p*(q-qb)-20*(abs(q)-abs(qb))))
   L=float(np.mean([g['settle'](q,x[0],p,p)['activation_adjustment']-g['settle'](qb,y[0],p,p)['activation_adjustment'] for x,y in zip(cs[:-1],bs[:-1])]))
   score=G-max(0,L);schedules[f'candidate_{k}B0_MW_each']=q
  rows.append(dict(k=k,status=int(meta['status']),qualification=ok,Ghat=G,Lhat=L,score=score,eligible=bool(ok and score>g['protocol']['positive_threshold'])))
 good=[r for r in rows if r['eligible']];best=max([r['score'] for r in good],default=0.)
 selected=min((r['k'] for r in good if r['score']>=best-g['protocol']['tie_tolerance']),default=1)
 out=g['O'];pd.DataFrame(rows).to_csv(out/'candidates.csv',index=False);pd.DataFrame(schedules).to_csv(out/'schedules.csv',index=False)
 report={'forecast_day':'2026-01-05','prediction_source':'freshly computed released LoRA inference','selected_k':selected,'reference_retained':not bool(good),'scope':'one-day planning and causal replay; no realized-price settlement or full AC evaluation'}
 (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
