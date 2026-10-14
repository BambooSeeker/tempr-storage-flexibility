from pathlib import Path
import json, shutil, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

R=Path.cwd(); O=R/'generated/figures';S=O/'source_data';S.mkdir(parents=True,exist_ok=True)
BLUE='#D85B3F'; ORANGE='#159B8E'; GRAY='#8A60A8'; INK='#28558C'; LIGHT='#B5CDE9'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],'font.size':8,
 'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
 'axes.linewidth':.65,'xtick.major.width':.6,'ytick.major.width':.6,
 'xtick.major.size':3,'ytick.major.size':3,'pdf.fonttype':42,'svg.fonttype':'none',
 'mathtext.fontset':'stix','savefig.dpi':600})
paths={'prices':'data/figure_sources/verified_market_prices_3840.csv',
 'daily':'results/experiments/reservation_v5/settled_daily.csv',
 'fixed8':'results/experiments/reservation_v8/settled_daily.csv',
 'candidates':'results/experiments/reservation_v5/candidates.csv',
 'selection':'results/experiments/reservation_v5/selection.csv',
 'network':'results/experiments/reservation_v5/AC/E2_online_AC_15min.csv'}
data={k:pd.read_csv(R/v) for k,v in paths.items()}
sel=data['selection'].query("method=='menu'").sort_values('day'); accepted=sel.loc[sel.accepted,'day'].to_numpy()
assert len(accepted)==20
records=[];issues=[]
def clean(ax):
 ax.spines[['top','right']].set_visible(False);ax.tick_params(direction='out',pad=3);ax.set_axisbelow(True)
def save(fig,name):
 fig.canvas.draw();ren=fig.canvas.get_renderer()
 for ax in fig.axes:
  for t in [ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels(),*ax.texts]:
   if t.get_visible() and t.get_text():
    b=t.get_window_extent(ren)
    if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5:issues.append([name,t.get_text()])
 fig.savefig(O/f'{name}.pdf',facecolor='white')
 fig.savefig(O/f'{name}.svg',facecolor='white')
 fig.savefig(O/f'{name}.png',facecolor='white',dpi=600)
 fig.savefig(O/f'{name}.tiff',facecolor='white',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
 records.append({'name':name,'width_in':fig.get_figwidth(),'height_in':fig.get_figheight()});plt.close(fig)

# Economic trajectory: each adjustment appears, including all negative returns.
daily=pd.concat([data['daily'].query("method in ['menu','fixed2']"),data['fixed8']],ignore_index=True)
wide=daily.pivot(index='day',columns='method',values='total_surplus').sort_index()
inactive=wide.loc[~wide.index.isin(accepted)]
assert abs(inactive.to_numpy()).max()<1e-6
events=wide.loc[accepted].copy();cum=events.cumsum()/1000
events.insert(0,'event',np.arange(1,21));events.to_csv(S/'Fig1_all_accepted_daily_returns.csv')
wide.to_csv(S/'Fig1_full_80day_returns.csv')
fig,ax=plt.subplots(figsize=(3.5,2.95));fig.subplots_adjust(left=.19,right=.97,bottom=.18,top=.79)
x=np.arange(21)
ax.axhline(0,color='#B3BBC1',lw=.55)
ax.plot(x,np.r_[0,cum.fixed8],color=GRAY,lw=1.1,ls='--',marker='o',ms=4.0,mfc='white',mec=GRAY,mew=.65,label='Fixed 8$B_0$',zorder=2)
ax.plot(x,np.r_[0,cum.fixed2],color=ORANGE,lw=1.15,marker='^',ms=2.5,mew=.5,label='Fixed 2$B_0$',zorder=3)
ax.plot(x,np.r_[0,cum.menu],color=BLUE,lw=1.3,marker='o',ms=2.0,mew=.5,label='Selected budget',zorder=4)
handles,labels=ax.get_legend_handles_labels();order=[2,1,0]
ax.legend([handles[i] for i in order],[labels[i] for i in order],loc='lower left',bbox_to_anchor=(0,1.025),
 frameon=False,ncol=2,handlelength=1.6,columnspacing=1.1,handletextpad=.45,borderaxespad=0)
ax.set_xlim(0,20.5);ax.set_xticks([0,5,10,15,20]);ax.set_ylim(-.5,2.65);ax.set_yticks([-.5,0,.5,1,1.5,2,2.5])
ax.set_xlabel('Allocation event');ax.set_ylabel('Cumulative joint gain (10³ CNY)')
clean(ax);save(fig,'Fig1_economic_trajectory')

# Mechanism: final three contiguous days, common clock in all panels.
pr=data['prices'].query('day==79').sort_values(['day','half_hour']);assert len(pr)==48
st=data['network'].query("day==79 and method in ['B2','menu']").sort_values(['method','day','slot']);assert len(st)==192
pr.to_csv(S/'Fig4_observed_and_predicted_prices_24h.csv',index=False);st.to_csv(S/'Fig4_computed_storage_paths_24h.csv',index=False)
fig,axs=plt.subplots(3,1,figsize=(3.5,4.2),sharex=True,gridspec_kw={'height_ratios':[1.05,1,1]})
fig.subplots_adjust(left=.21,right=.98,bottom=.12,top=.92,hspace=.52)
edge=np.arange(97)/4
axs[0].stairs(np.repeat(pr.observed_DA_CNY_MWh.to_numpy(),2),edge,baseline=None,color=INK,lw=1.05,label='Observed DA price')
axs[0].stairs(np.repeat(pr.LoRA_forecast_CNY_MWh.to_numpy(),2),edge,baseline=None,color=BLUE,lw=1,ls='--',label='LoRA forecast')
axs[0].set_ylim(-420,450);axs[0].set_yticks([-400,0,400]);axs[0].set_ylabel('Price\n(CNY/MWh)',labelpad=6)
axs[0].legend(loc='lower right',bbox_to_anchor=(1,1.10),frameon=False,ncol=2,handlelength=1.4,columnspacing=.8,borderaxespad=0)
for method,col,ls,label in [('B2',INK,'--','Fixed rights'),('menu',BLUE,'-','Selected budget')]:
 a=st.query('method==@method').sort_values(['day','slot'])
 assert abs(a.groupby('day').soc.last()-.5).max()<1e-7
 axs[1].stairs(a.q_DA_each.to_numpy(),edge,baseline=None,color=col,lw=1,ls=ls,label=label)
 axs[2].plot(edge,np.r_[.5,a.soc.to_numpy()]*100,color=col,lw=1,ls=ls)
axs[1].set_ylim(-2.4,2.4);axs[1].set_yticks([-2,0,2]);axs[1].axhline(0,color=LIGHT,lw=.5)
axs[1].set_ylabel('Injection\n(MW/battery)',labelpad=6)
axs[1].legend(loc='lower right',bbox_to_anchor=(1,1.02),frameon=False,ncol=2,handlelength=1.4,columnspacing=.8,borderaxespad=0)
axs[2].set_ylim(0,100);axs[2].set_yticks([10,50,90]);axs[2].set_ylabel('SOC (%)',labelpad=6)
for bound in [10,90]:axs[2].axhline(bound,color=LIGHT,lw=.6,ls=':')
for i,ax in enumerate(axs):
 ax.text(-.23,1.04,chr(97+i),transform=ax.transAxes,fontsize=9,fontweight='bold',va='bottom')

 ax.set_xlim(0,24);clean(ax)
axs[2].set_xticks([0,6,12,18,24]);axs[2].set_xlabel('Time of day (h)')
axs[2].get_xticklabels()[0].set_ha('left');axs[2].get_xticklabels()[-1].set_ha('right')
save(fig,'Fig4_daily_mechanism')

result={'figures':records,'layout_issues':issues,'multipanel_figures':2,'event_days':accepted.tolist(),
 'inactive_days':60,'max_inactive_daily_gain':float(abs(inactive.to_numpy()).max()),
 'total_joint_gain_CNY':wide.sum().to_dict(),'input_hashes':[{'path':v,'sha256':hashlib.sha256((R/v).read_bytes()).hexdigest()} for v in paths.values()]}
(O/'figure_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');shutil.copy2(__file__,O/Path(__file__).name)
print(json.dumps({'issues':issues,'totals':result['total_joint_gain_CNY']},indent=2))
