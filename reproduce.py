"""Single entry point; code checks, live LoRA figure demo, or frozen paper reproduction."""
from pathlib import Path
import argparse,subprocess,sys,os,ast,hashlib,json,time
R=Path(__file__).resolve().parent
def run(script):
 env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['MPLBACKEND']='Agg'
 print('[RUN]',script,flush=True);subprocess.run([sys.executable,str(R/script)],cwd=R,env=env,check=True)
def check():
 count=0
 for base in ['src','scripts','forecast/code','tests']:
  for p in (R/base).rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p));count+=1
 for name in ['demo.py','demo_market.py','infer_figures.py','reproduce.py','forecast/run_reproduce.py','forecast/run_corrected_baselines.py']:ast.parse((R/name).read_text(encoding='utf-8-sig'));count+=1
 import csv
 with (R/'manifest.csv').open(encoding='utf-8',newline='') as f:
  rows=list(csv.DictReader(f))
 for item in rows:
  p=R/item['path'];assert p.is_file(),item['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],item['path']
 print('Syntax:',count,'Python files; verified hashes:',len(rows))
def main():
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--demo',action='store_true');p.add_argument('--full',action='store_true');p.add_argument('--no-figures',action='store_true');a=p.parse_args();os.chdir(R);tic=time.monotonic();check()
 if a.check and not (a.demo or a.full):return
 if a.demo or not a.full:run('demo.py');return
 if not (R/'forecast/data').exists() or not (R/'results/experiments/reservation_v5/selection.csv').exists():
  raise SystemExit('Full study records are not included in the public example. See DATA_AVAILABILITY.md and docs/REPRODUCING.md.')
 (R/'generated/figures').mkdir(parents=True,exist_ok=True)
 run('scripts/verify_ledgers.py');run('scripts/rebuild_tables.py')
 if not a.no_figures:
  for s in ['plot_forecasts','plot_allocation','plot_service','plot_value']:run('scripts/'+s+'.py')
  import shutil
  mapping={'Forecast_month':2,'Forecast_week':3,'Forecast_ablation':4,'Forecast_operating_states':5,'Fig8_typical_import_days':6,'Fig1_economic_trajectory':7,'Fig7_typical_budget_responses':8,'Fig4_daily_mechanism':9}
  for old,num in mapping.items():
   for ext in ['png','pdf','svg']:
    src=R/f'generated/figures/{old}.{ext}';shutil.copy2(src,R/f'generated/figures/Fig{num:02}.{ext}')
  shutil.copy2(R/'artwork/Fig1_framework.png',R/'generated/figures/Fig01.png')
 (R/'generated/run_summary.json').write_text(json.dumps({'mode':'frozen-record reproduction','seconds':time.monotonic()-tic,'new_training':False,'new_full_AC_run':False},indent=2))
 print('Finished:',R/'generated')
if __name__=='__main__':main()
