"""Current-state AC feedback only; DA plans/contracts remain frozen."""
from pathlib import Path
import json,time
import numpy as np,pandas as pd
import pandapower as pp,simbench as sb
from threadpoolctl import threadpool_limits
threadpool_limits(1)
O=Path('results/experiments/reservation_v5/AC');W=Path('data/network');prof=np.load(W/'network_profiles.npz');l=prof['load'];ql=prof['qload'];g=prof['sgen'];z=prof['net']
cfg=json.loads((O/'pilot_manifest.json').read_text(encoding='utf-8'));tr=pd.read_csv(O/'E2_pilot_traces.csv');daily=pd.read_csv(O/'E2_pilot_daily.csv');P=cfg['P_each_MW'];EM=cfg['E_each_MWh'];eta=np.sqrt(.85);dt=.25;cap=cfg['import_cap_MW'];B=cfg['budget_hours']
base=Path(r'forecast');pr=pd.read_csv(base/'outputs/daily_pred_Chronos2_ExogDirect_LoRA600.csv');starts=pd.to_datetime(pr.iloc[:,0]);y=pd.read_csv(base/'outputs/y_test.csv').set_index('forecast_start_time').reindex(pr.iloc[:,0]);DA=y.to_numpy()
frames=[]
for f in (base/'data').glob('*.csv'):
    d=pd.read_csv(f,usecols=['time','code','price_real']);frames.append(d[d.code.eq('YQFDC-01')])
d=pd.concat(frames);d.time=pd.to_datetime(d.time);RT=d.set_index('time').reindex([t+pd.Timedelta(minutes=30*k) for t in starts for k in range(48)]).price_real.to_numpy().reshape(80,48)
n=pp.from_json('data/network/network.json');orig=n.sgen.index.copy();bs=[pp.create_sgen(n,bus,0,q_mvar=0,name='storage') for bus in cfg['battery_buses']]
def flow(x):
    n.sgen.loc[bs,'p_mw']=x;pp.runpp(n,numba=False,init='results' if n.converged else 'auto');return float(n.res_ext_grid.p_mw.sum())
rows=[];points=[];tic=time.monotonic()
for method in ['menu','fixed2']:
    qs=tr[tr.method.eq(method)].q_DA_each.to_numpy().reshape(80,96);dr=daily[daily.method.eq(method)].set_index('day');e=.5*EM
    for day in range(80):
        qq=qs[day];r=float(dr.loc[day,'r_each_MW']);budget=float(dr.loc[day,'budget_each_MWh']);eb=np.r_[e,e+np.cumsum((eta*np.maximum(-qq,0)-np.maximum(qq,0)/eta)*dt)];used=0.;xx=[];pcc=[];em=[];act=[];maxfailure=0.;thermal=voltage=fail=0
        for t in range(96):
            i=(286+day)*96+t;n.load.loc[:,'p_mw']=l[i];n.load.loc[:,'q_mvar']=ql[i];n.sgen.loc[orig,'p_mw']=g[i]
            a=0.;x=qq[t]
            if 88<=t:
                gap=max(0,eb[t]-e);fbase=eta*max(-qq[t],0)-max(qq[t],0)/eta;ftarget=fbase+gap/dt;x=(-ftarget/eta if ftarget>=0 else -ftarget*eta);x=max(-P,min(qq[t],x))
            ppcc=flow(x)
            if 24<=t<88:
                for it in range(3):
                    if ppcc<=cap+1e-7:break
                    upper=min(r,max(0,budget-used)/dt,max(0,(e-.1*EM)*eta/dt-qq[t]),P-qq[t])
                    anew=min(upper,a+max(0,(ppcc-cap)/2)*1.01+1e-5)
                    if anew<=a+1e-10:break
                    a=anew;x=qq[t]+a;ppcc=flow(x)
                used+=a*dt
            elif 88<=t:
                for it in range(3):
                    if ppcc<=cap+1e-7 or x>=qq[t]-1e-10:break
                    x=min(qq[t],x+max(0,(ppcc-cap)/2)*1.01+1e-5);ppcc=flow(x)
            e+=(eta*max(-x,0)-max(x,0)/eta)*dt
            vmlo=float(n.res_bus.vm_pu.min());vmhi=float(n.res_bus.vm_pu.max());line=float(n.res_line.loading_percent.max());trafo=float(n.res_trafo.loading_percent.max());loss=float(n.res_line.pl_mw.sum()+n.res_trafo.pl_mw.sum());excess=max(0,ppcc-cap)
            voltage+=int(vmlo<.95 or vmhi>1.05);thermal+=int(line>100 or trafo>100)
            xx.append(x);pcc.append(ppcc);em.append(e);act.append(a);maxfailure=max(maxfailure,excess)
            points.append({'method':method,'day':day,'slot':t,'q_DA_each':qq[t],'q_exec_each':x,'call_each':a,'soc':e/EM,'pcc_mw':ppcc,'loss_MW':loss,'cap_excess_MW':excess,'vm_min':vmlo,'vm_max':vmhi,'line_max':line,'trafo_max':trafo})
        xx=np.array(xx);pda=np.repeat(DA[day],2);prt=np.repeat(RT[day],2)
        R=2*dt*np.sum(pda*qq+prt*(xx-qq));deg=2*20*dt*np.sum(abs(xx));Rbase=2*dt*np.sum(pda*qq);dbase=2*20*dt*np.sum(abs(qq));S=-(R-Rbase)+(deg-dbase);tau=float(dr.loc[day,'refund']);deficit=max(0,eb[-1]-e)*2
        rows.append({'day':day,'method':method,'r_each':r,'refund':tau,'net_owner_excl_F':R-deg+S-tau,'grid_contract_cost_excl_F':S-tau,'energy_revenue':R,'degradation':deg,'activation_adjustment':S,'activation_MWh':2*used,'budget_violation_MWh':max(0,used-budget),'max_cap_excess_MW':maxfailure,'cap_excess_MWh':sum(max(0,k-cap) for k in pcc)*dt,'recovery_deficit_MWh':deficit,'terminal_soc':e/EM,'soc_min':min(em)/EM,'soc_max':max(em)/EM,'voltage_violations':voltage,'thermal_violations':thermal,'full_service_pass':maxfailure<1e-6 and deficit<1e-6 and voltage==0 and thermal==0 and min(em)>=.1*EM-1e-7 and max(em)<=.9*EM+1e-7})
        if (day+1)%10==0:
            pd.DataFrame(rows).to_csv(O/'E2_online_AC_daily.csv',index=False);pd.DataFrame(points).to_csv(O/'E2_online_AC_15min.csv',index=False);print('online AC',method,day+1,'s',round(time.monotonic()-tic,1),flush=True)
pd.DataFrame(rows).to_csv(O/'E2_online_AC_daily.csv',index=False)
pd.DataFrame(points).to_csv(O/'E2_online_AC_15min.csv',index=False)
summary={}
for method,a in pd.DataFrame(rows).groupby('method'):
    summary[method]={'days':len(a),'full_service_pass_days':int(a.full_service_pass.sum()),'cap_excess_MWh':float(a.cap_excess_MWh.sum()),'recovery_deficit_MWh':float(a.recovery_deficit_MWh.sum()),'net_owner_excl_F':float(a.net_owner_excl_F.sum()),'grid_contract_cost_excl_F':float(a.grid_contract_cost_excl_F.sum()),'max_budget_violation_MWh':float(a.budget_violation_MWh.max()),'terminal_soc':float(a.iloc[-1].terminal_soc),'voltage_violation_points':int(a.voltage_violations.sum()),'thermal_violation_points':int(a.thermal_violations.sum())}
(O/'E2_online_AC_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)
