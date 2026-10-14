"""Run archived experimental algorithms in an isolated directory."""
from pathlib import Path
import argparse,subprocess,sys,shutil,os,json,time
R=Path(__file__).resolve().parents[1]
CHAINS={
 'main':['round5_plan','run_round5_ac_full','close_round5_ledger'],
 'fixed8':['round8_fixed8_plan','ac8_fixed8','ledger8_fixed8'],
 'pressure':['round6_plan','round6_ac_batch'],
 'unfinetuned':['round7_forecast_plan','ac7_unfinetuned','ledger7_unfinetuned'],
 'strong':['round7_strong','ac7_strong','ledger7_strong']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',choices=list(CHAINS)+['all'],default='main');p.add_argument('--run-dir',default='runs/market');a=p.parse_args()
 dest=Path(a.run_dir).resolve();
 if dest.exists():raise SystemExit('Choose a new --run-dir; existing runs are preserved.')
 for rel in ['forecast/data','forecast/outputs','data/network']:
  if not (R/rel).exists():raise SystemExit('Missing authorized inputs: '+rel+'; see docs/REPRODUCING.md')
 dest.mkdir(parents=True)
 for rel in ['src','config','forecast/data','forecast/outputs','data/network']:
  shutil.copytree(R/rel,dest/rel,ignore=shutil.ignore_patterns('__pycache__'))
 cases=list(CHAINS) if a.case=='all' else [a.case]
 # Secondary cases consume the documented frozen main-study reference.
 if a.case not in ['main','all']:
  src=R/'results/experiments/reservation_v5'
  if not src.exists():raise SystemExit('Secondary cases require reservation_v5 records or run --case all.')
  shutil.copytree(src,dest/'results/experiments/reservation_v5')
 for rel in ['results/experiments','results/experiments/reservation_v7']:(dest/rel).mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';tic=time.monotonic();done=[]
 for case in cases:
  for script in CHAINS[case]:
   print('[RUN]',case,script,flush=True)
   with (dest/(script+'.log')).open('w',encoding='utf-8') as f:
    subprocess.run([sys.executable,str(dest/'src/market'/f'{script}.py')],cwd=dest,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
   done.append(script)
 (dest/'execution.json').write_text(json.dumps({'completed':done,'seconds':time.monotonic()-tic,'new_training':False,'new_AC':True},indent=2))
 print('New physical ledgers:',dest/'results/experiments')
if __name__=='__main__':main()
