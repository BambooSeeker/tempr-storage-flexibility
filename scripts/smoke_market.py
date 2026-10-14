"""One recorded optimization and one AC state, without overwriting frozen results."""
from pathlib import Path
import os,json
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];os.chdir(R)
s=(R/'src/market/round5_plan.py').read_text(encoding='utf-8').split('rows=[];selected=[];traces=[];tic=')[0]
g={};exec(compile(s,'planner_setup','exec'),g)
day=0;p=np.repeat(g['pred'][day],2);sc=np.vstack([g['mean'][(286+day)%7]+g['res'][g['cfg']['calibration_residual_ids']]+g['loss_margin'],g['U']])
qb,meta=g['solve_flexible'](p,sc,g['r0'],g['e0']);assert qb is not None
result,meta=g['optimize'](p,sc,'qualified',budget_cap=2*g['B0']);assert result is not None
G=2*g['dt']*np.sum(p*(result['q']-qb)-20*(abs(result['q'])-abs(qb)))
ref=pd.read_csv(R/'results/experiments/reservation_v5/candidates.csv').query('day==0 and k==2').iloc[0];assert abs(G-ref.Ghat)<1e-4,(G,ref.Ghat)
import pandapower as pp
net=pp.from_json(str(R/'data/network/network.json'));z=np.load(R/'data/network/network_profiles.npz');original=net.sgen.index.copy();row=pd.read_csv(R/'results/experiments/reservation_v5/AC/E2_online_AC_15min.csv').query("method=='menu' and day==0 and slot==0").iloc[0]
idx=286*96;net.load.loc[:,'p_mw']=z['load'][idx];net.load.loc[:,'q_mvar']=z['qload'][idx];net.sgen.loc[original,'p_mw']=z['sgen'][idx]
for bus in g['cfg']['battery_buses']:pp.create_sgen(net,bus,float(row.q_exec_each),q_mvar=0,name='storage')
pp.runpp(net,numba=False);err=abs(float(net.res_ext_grid.p_mw.sum())-row.pcc_mw);assert err<1e-5,err
out=R/'generated';out.mkdir(exist_ok=True);(out/'market_smoke.json').write_text(json.dumps({'optimization_gain_error_CNY':float(abs(G-ref.Ghat)),'PCC_error_MW':err,'scope':'one optimization day and one AC operating point'},indent=2));print('Optimization and AC smoke checks passed.')
