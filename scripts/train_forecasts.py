"""Train the original model set in an isolated copy; requires full authorized inputs."""
from pathlib import Path
import argparse,shutil,subprocess,sys,os,json,time
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',default='runs/forecast');a=p.parse_args();d=Path(a.run_dir).resolve()
 if d.exists():raise SystemExit('Use a new run directory to preserve earlier records.')
 if not list((R/'forecast/data').glob('*.csv')):raise SystemExit('Obtain authorized complete forecast/data inputs; the public demo is not a training dataset.')
 shutil.copytree(R/'forecast',d/'forecast',ignore=shutil.ignore_patterns('outputs','__pycache__'));(d/'forecast/outputs').mkdir()
 # Read only the script list; importing the legacy runner does not execute experiments.
 import importlib.util
 spec=importlib.util.spec_from_file_location('original_runner',R/'forecast/run_reproduce.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';tic=time.monotonic()
 for name in m.MODEL_SCRIPTS:
  print('[TRAIN/EVALUATE]',name,flush=True)
  with (d/(name+'.log')).open('w',encoding='utf-8') as log:subprocess.run([sys.executable,str(d/'forecast/code'/name)],cwd=d/'forecast/code',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 (d/'execution.json').write_text(json.dumps({'new_training':True,'seconds':time.monotonic()-tic,'scripts':m.MODEL_SCRIPTS},indent=2))
 print('New model outputs:',d/'forecast/outputs')
if __name__=='__main__':main()
