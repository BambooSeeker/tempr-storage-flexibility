"""Run the released paper LoRA adapter and regenerate Fig. 2 / Fig. 3 windows."""
from pathlib import Path
import argparse,hashlib,json,os,time
os.environ.setdefault('USE_TF','0');os.environ.setdefault('USE_TORCH','1')
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent

def load_inputs():
 c=json.loads((R/'config/figure_demo.json').read_text(encoding='utf-8'));D=R/'data/figures'
 h=pd.read_csv(D/'initial_history.csv',parse_dates=['time']);x=pd.read_csv(D/'operating_inputs.csv',parse_dates=['time'])
 assert len(h)==1024 and len(x)==672 and 'price_day_ahead' not in x
 assert list(x.columns)==['time']+c['features']
 t=pd.concat([h.time,x.time],ignore_index=True);assert t.diff().dropna().eq(pd.Timedelta(minutes=30)).all()
 for frame in [h,x]:assert np.isfinite(frame.drop(columns='time').to_numpy()).all()
 return c,h,x

def make_task(history,future,features):
 assert len(history)==1024 and len(future)==48
 assert history.time.max()<future.time.min()
 return {'target':history.price_day_ahead.to_numpy(np.float32),
         'past_covariates':{k:history[k].to_numpy(np.float32) for k in features},
         'future_covariates':{k:future[k].to_numpy(np.float32) for k in features}}

def load_model(c,args):
 import torch
 from chronos import Chronos2Pipeline
 from peft import PeftModel
 adapter=R/'models/chronos2_lora'
 assert hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()==c['adapter_sha256']
 torch.set_num_threads(args.threads)
 # The original training code loaded the base in BF16, then fit() copied those
 # rounded weights into a new FP32 model before adding LoRA. Preserve that path.
 base=Chronos2Pipeline.from_pretrained(args.base_model or c['base_model'],revision=None if args.base_model else c['base_revision'],device_map=args.device,torch_dtype=torch.bfloat16,local_files_only=args.offline)
 base.model.float()
 adapted=PeftModel.from_pretrained(base.model,str(adapter),is_trainable=False)
 adapted.to(torch.bfloat16 if args.precision=="bfloat16" else torch.float32)
 adapted.eval()
 return Chronos2Pipeline(model=adapted)

def render(pred,c,out):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import matplotlib.dates as mdates
 from matplotlib.lines import Line2D
 # Comparator results are read only after all new LoRA predictions are complete.
 baseline=pd.read_csv(R/'data/figures/published_comparators.csv',parse_dates=['time'])
 assert pred.time.equals(baseline.time)
 a=pred.merge(baseline,on='time',validate='one_to_one')
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'axes.labelsize':8.5,'legend.fontsize':8,'legend.frameon':False,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
 summary={}
 for name,window in c['figures'].items():
  start=pd.Timestamp(window['start']);end=start+pd.Timedelta(days=window['days']);b=a[(a.time>start)&(a.time<=end)].copy();assert len(b)==336
  fig,ax=plt.subplots(figsize=(3.5,2.7));fig.subplots_adjust(left=.19,right=.97,bottom=.20,top=.72)
  other=[k for k in baseline if k not in ['time','LightGBM','Chronos2_ExogDirect']]
  for k in other:ax.plot(b.time,b[k],color='#B5CDE9',lw=.35,alpha=.65,zorder=1)
  ax.plot(b.time,b.LightGBM,color='#8A60A8',lw=.65,zorder=2)
  ax.plot(b.time,b.Chronos2_ExogDirect,color='#159B8E',lw=.8,zorder=3)
  ax.plot(b.time,b.observed,color='#28558C',lw=.85,zorder=4)
  ax.plot(b.time,b.LoRA_live,color='#D85B3F',lw=1,zorder=5)
  ax.set_xlim(start,end);ax.set_ylabel('Price (CNY/MWh)');ax.set_xticks(pd.date_range(start+pd.Timedelta(days=1),periods=3,freq='2D'));ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
  handles=[Line2D([],[],color='#28558C',lw=.85,label='Observed'),Line2D([],[],color='#D85B3F',lw=1,label='Chronos-2 + OV + LoRA'),Line2D([],[],color='#159B8E',label='Chronos-2 + OV'),Line2D([],[],color='#8A60A8',label='LightGBM'),Line2D([],[],color='#B5CDE9',label=f'Other models ({len(other)})')]
  ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(-.03,1.025),ncol=2,handlelength=1.1,columnspacing=.6,handletextpad=.4,borderaxespad=0)
  for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}',dpi=600)
  plt.close(fig);b.to_csv(out/f'{name}_source.csv',index=False)
  e=b.LoRA_live.to_numpy()-b.observed.to_numpy();y=b.observed.to_numpy()
  summary[name]={'points':len(b),'MAE':float(np.abs(e).mean()),'RMSE':float(np.sqrt(np.mean(e**2))),'MAPE_percent':float(np.mean(abs(e[y!=0]/y[y!=0]))*100)}
 return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4);p.add_argument('--precision',choices=['bfloat16','float32'],default='bfloat16');p.add_argument('--base-model',help='Optional local path to the pinned base checkpoint');p.add_argument('--offline',action='store_true');a=p.parse_args()
 c,h,x=load_inputs();tic=time.monotonic();print('Loading original paper adapter ...',flush=True);pipeline=load_model(c,a)
 # Replay the information clock day by day: reveal today's labels only AFTER
 # predicting today's 48 points, then append them to tomorrow's historical context.
 rows=[];from_torch=__import__('torch')
 for day in range(14):
  future=x.iloc[day*48:(day+1)*48].copy();task=make_task(h.tail(1024),future,c['features'])
  with from_torch.inference_mode():
   _,means=pipeline.predict_quantiles(inputs=[task],prediction_length=48,batch_size=100,context_length=2048,cross_learning=False,limit_prediction_length=False)
  forecast=means[0][0].detach().cpu().numpy().astype(float)
  # This load occurs after inference. No future label is passed to the model.
  observed=pd.read_csv(R/'data/figures/observations.csv',parse_dates=['time']).iloc[day*48:(day+1)*48].copy()
  assert future.time.reset_index(drop=True).equals(observed.time.reset_index(drop=True))
  rows.append(pd.DataFrame({'time':future.time.to_numpy(),'observed':observed.price_day_ahead.to_numpy(),'LoRA_live':forecast}))
  h=pd.concat([h,future.merge(observed,on='time',validate='one_to_one')],ignore_index=True).tail(1024)
  print(f'Predicted {day+1}/14 days',flush=True)
 out=R/'generated/live_figures';out.mkdir(parents=True,exist_ok=True);pred=pd.concat(rows,ignore_index=True);pred.to_csv(out/'live_predictions.csv',index=False)
 metrics=render(pred,c,out)
 report={'prediction_source':'fresh LoRA model inference','comparator_source':'published 14-day baseline predictions','adapter_sha256':c['adapter_sha256'],'base_revision':c['base_revision'],'device':a.device,'precision':a.precision,'seconds':time.monotonic()-tic,'metrics':metrics}
 (out/'run.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
