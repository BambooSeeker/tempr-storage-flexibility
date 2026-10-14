from pathlib import Path
import pandas as pd,numpy as np,json,shutil,hashlib
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
R=Path.cwd();D=R/'generated';S=D/'figure_sources';S.mkdir(parents=True,exist_ok=True);(D/'figures').mkdir(exist_ok=True)

a=pd.read_csv(R/'data/figure_sources/all_80day_forecasts.csv');a.time=pd.to_datetime(a.time)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],'font.size':8,'axes.labelsize':8.5,'legend.fontsize':8,'legend.frameon':False,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
for name,days in [('Forecast_month',7),('Forecast_week',7)]:
 start=pd.Timestamp('2026-01-12' if name=='Forecast_month' else '2026-01-05');end=start+pd.Timedelta(days=days);b=a[(a.time>start)&(a.time<=end)];assert len(b)==48*days;b.to_csv(S/f'{name}.csv',index=False)
 fig,ax=plt.subplots(figsize=(3.5,2.7));fig.subplots_adjust(left=.19,right=.97,bottom=.20,top=.72)
 main=['LightGBM','Chronos2_ExogDirect','Chronos2_ExogDirect_LoRA600'];other=[c for c in b.columns if c not in ['time','observed']+main]
 for c in other:ax.plot(b.time,b[c],color='#B5CDE9',lw=.35,alpha=.65,zorder=1)
 ax.plot(b.time,b.LightGBM,color='#8A60A8',lw=.65,ls='-',zorder=2)
 ax.plot(b.time,b.Chronos2_ExogDirect,color='#159B8E',lw=.8,ls='-',zorder=3)
 ax.plot(b.time,b.observed,color='#28558C',lw=.85,zorder=4)
 ax.plot(b.time,b.Chronos2_ExogDirect_LoRA600,color='#D85B3F',lw=1,zorder=5)
 ax.set_xlim(start,end);ax.set_ylabel('Price (CNY/MWh)')
 if days==7:
  ax.set_xticks(pd.date_range(start+pd.Timedelta(days=1),periods=3,freq='2D'));ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
 else:
  ax.set_xticks(pd.date_range(start,end,freq='24h'));ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'));ax.get_xticklabels()[0].set_ha('left');ax.get_xticklabels()[-1].set_ha('right')
 handles=[Line2D([],[],color='#28558C',lw=.85,label='Observed'),Line2D([],[],color='#D85B3F',lw=1,label='Chronos-2 + OV + LoRA'),Line2D([],[],color='#159B8E',ls='-',label='Chronos-2 + OV'),Line2D([],[],color='#8A60A8',ls='-',label='LightGBM'),Line2D([],[],color='#B5CDE9',label=f'Other models ({len(other)})')]
 ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(-.03,1.025),ncol=2,handlelength=1.1,columnspacing=.6,handletextpad=.4,borderaxespad=0)
 f=D/'figures'/name;fig.savefig(str(f)+'.pdf');fig.savefig(str(f)+'.svg');fig.savefig(str(f)+'.png',dpi=600);fig.savefig(str(f)+'.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'});plt.close(fig)
