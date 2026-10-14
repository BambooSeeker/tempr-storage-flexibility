from pathlib import Path
import json,shutil,pandas as pd,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path.cwd();D=R/'generated';D.mkdir(exist_ok=True)
S=D/'figure_sources';S.mkdir(exist_ok=True)
E=R/'data/figure_sources';a=pd.read_csv(E/'paired_80_days.csv');g=pd.read_csv(E/'daily_summary.csv').set_index('day');summary=json.loads((E/'summary.json').read_text(encoding='utf-8'));prices=pd.read_csv(R/'data/figure_sources/verified_market_prices_3840.csv');dates=pd.to_datetime(prices.groupby('day').time.first()).dt.normalize();days=[46,57,59,62,72,73]
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],'font.size':8,'axes.labelsize':8.5,'legend.fontsize':8,'legend.frameon':False,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axs=plt.subplots(3,2,figsize=(3.5,4.15),sharex=True,sharey=True);fig.subplots_adjust(left=.15,right=.98,bottom=.12,top=.84,wspace=.14,hspace=.52)
edge=np.arange(97)/4;cap=summary['cap_MW']
for ax,day in zip(axs.flat,days):
 b=a[a.day.eq(day)].sort_values('slot');assert len(b)==96
 ax.stairs(b.pcc_mw,edge,baseline=None,color='#D85B3F',lw=.85,zorder=2)
 ax.stairs(b.pcc_plan_only_MW,edge,baseline=None,color='#28558C',lw=.7,ls='--',zorder=3)
 ax.axhline(cap,color='#202020',lw=.6,ls=':',zorder=1)
 xx=b[b.call_each>1e-8]
 if len(xx):ax.scatter((xx.slot+.5)/4,xx.pcc_plan_only_MW,s=12,facecolors='none',edgecolors='#28558C',linewidths=.7,zorder=4)
 ax.set(xlim=(0,24),ylim=(0,20),xticks=[0,12,24],yticks=[0,10,20]);ax.set_title(dates.loc[day].strftime('%d %b %Y')+f' | {g.loc[day,"activation_MWh"]:.3f} MWh',fontsize=7.5,pad=4);ax.tick_params(length=2,pad=2)
fig.legend(handles=[Line2D([],[],color='#28558C',lw=.8,ls='--',label='Trading schedule only'),Line2D([],[],color='#D85B3F',lw=.9,label='With service delivery'),Line2D([],[],color='#202020',lw=.6,ls=':',label='Import ceiling')],loc='upper center',bbox_to_anchor=(.56,1.005),ncol=1,labelspacing=.25,handlelength=1.8)
fig.supylabel('PCC import (MW)',x=.015,fontsize=8.5);fig.supxlabel('Time of day (h)',y=.015,fontsize=8.5)
f=D/'figures/Fig8_typical_import_days';fig.savefig(str(f)+'.pdf');fig.savefig(str(f)+'.svg');fig.savefig(str(f)+'.png',dpi=600);fig.savefig(str(f)+'.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
for name in ['paired_80_days.csv','daily_summary.csv','summary.json']:shutil.copy2(E/name,S/name)
print('Fig8 paired comparison rendered.')
