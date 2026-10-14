from pathlib import Path
import json,shutil
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
R=Path.cwd();D=R/'generated';O=D/'figures';S=D/'figure_sources';S.mkdir(exist_ok=True)
GREEN='#8A60A8';GOLD='#159B8E';ORANGE='#D85B3F';INK='#202020';GRAY='#28558C'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],'font.size':8,'axes.labelsize':8.5,'legend.fontsize':8,'legend.frameon':False,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'stix'})
def save(fig,name):
 fig.savefig(O/(name+'.pdf'));fig.savefig(O/(name+'.svg'));fig.savefig(O/(name+'.png'),dpi=600);fig.savefig(O/(name+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'});plt.close(fig)
a=pd.read_csv(R/'data/figure_sources/Forecast_ablation.csv');a.to_csv(S/'Forecast_ablation.csv',index=False)
fig,ax=plt.subplots(figsize=(3.5,2.05));fig.subplots_adjust(left=.34,right=.97,bottom=.25,top=.96)
ax.barh(range(4),a.mae,height=.58,color=[GRAY,GREEN,GOLD,ORANGE],zorder=3);ax.set_yticks(range(4),['Price only','Price residual','Operating inputs','Inputs + LoRA']);ax.invert_yaxis();ax.set_xlim(0,28);ax.set_xticks([0,5,10,15,20,25]);ax.set_xlabel('MAE (CNY/MWh)');ax.tick_params(axis='y',length=0)
for i,v in enumerate(a.mae):ax.text(v+.45,i,f'{v:.2f}',va='center')
save(fig,'Forecast_ablation')
b=pd.read_csv(R/'data/figure_sources/Forecast_operating_states.csv');b.to_csv(S/'Forecast_operating_states.csv',index=False)
models=['LightGBM','Chronos2 + target-day variables','LoRA-Chronos2'];conds=['Demand','Renewable penetration','Residual bidding space'];v=np.array([[b[(b.condition==c)&(b.tertile==t)&(b.model==m)].mae.item() for m in models] for c in conds for t in ['T1','T2','T3']])
fig,ax=plt.subplots(figsize=(3.5,3.25));fig.subplots_adjust(left=.35,right=.97,bottom=.17,top=.80);y=np.array([0,1,2,3.4,4.4,5.4,6.8,7.8,8.8])
for j,(label,col,hatch) in enumerate(zip(['LightGBM','Chronos-2 + OV','LoRA-Chronos-2'],[GREEN,GOLD,ORANGE],['','//',''])):ax.barh(y+(j-1)*.24,v[:,j],height=.21,color=col,edgecolor='white',linewidth=.25,hatch=hatch,label=label,zorder=3)
ax.set_yticks(y,[f'{c}  {t}' for c in ['Demand','Renewables','Residual space'] for t in ['L','M','H']]);ax.invert_yaxis();ax.set_xlim(0,30);ax.set_xticks([0,10,20,30]);ax.set_xlabel('MAE (CNY/MWh)');ax.tick_params(axis='y',length=0,pad=4);ax.legend(loc='lower left',bbox_to_anchor=(-.02,1.025),borderaxespad=0,handlelength=1.6,labelspacing=.3);save(fig,'Forecast_operating_states')
P=R/'results/experiments/reservation_v5';sel=pd.read_csv(P/'selection.csv').query("method=='menu'");c=pd.read_csv(P/'candidates.csv');accepted=sel.query('accepted');wide=c.pivot(index='day',columns='k',values='score');ids=accepted.day.to_list();norm=wide.loc[ids,[1,2,4,8]].div(wide.loc[ids,[1,2,4,8]].max(axis=1),axis=0);assert np.isfinite(norm).all().all();norm.to_csv(S/'Fig7_normalized_scores.csv');c.to_csv(S/'Fig7_all_candidate_scores.csv',index=False);sel.to_csv(S/'Fig7_all_selections.csv',index=False)
fig,ax=plt.subplots(figsize=(3.5,2.85));fig.subplots_adjust(left=.19,right=.97,bottom=.20,top=.78)
for k,col,mark,ls in [(2,GOLD,'s','-'),(4,GRAY,'o','--'),(8,GREEN,'^','-.')]:
 group=accepted.loc[accepted.selected_k.eq(k),'day']
 for day in group:
  row=wide.loc[day,[1,2,4,8]]
  ax.plot([0,1,2,3],row.values,color=col,lw=.75,alpha=.8,ls=ls,marker=mark,ms=2.2,mfc='white',mew=.5)
  ax.scatter([[1,2,4,8].index(k)],[row.loc[k]],s=18,c=col,marker=mark,zorder=4)
ax.set_yscale('symlog',linthresh=10,linscale=.7)
ax.set(xlim=(-.12,3.12),ylim=(-.8,2300),xticks=[0,1,2,3],yticks=[0,10,100,1000],xlabel=r'Candidate energy budget ($B_0$)',ylabel='Predicted allocation score (CNY)')
ax.set_xticklabels(['1','2','4','8']);ax.set_yticklabels(['0','10','100','1,000']);ax.minorticks_off()
ax.legend(handles=[Line2D([],[],color=col,ls=ls,marker=mark,ms=3,lw=.8,label=rf'{k}$B_0$ (n = {n})') for k,col,mark,ls,n in [(2,GOLD,'s','-',10),(4,GRAY,'o','--',9),(8,GREEN,'^','-.',1)]],loc='lower left',bbox_to_anchor=(0,1.025),borderaxespad=0,ncol=2,handlelength=1.6,columnspacing=.8)
save(fig,'Fig7_typical_budget_responses')
