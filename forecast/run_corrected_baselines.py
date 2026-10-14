"""Retrain the five affected comparators on purged target windows.

Requires authorized monthly data. Results are written only to --output-dir.
The test origins and all model hyperparameters are retained.
"""
import argparse,os,sys,time,json,subprocess,platform
from pathlib import Path

MODELS=['LightGBM','SVR_PriceLags','LSTM_ResidualDelta','Transformer_ResidualDelta','PatchTST_ResidualDelta']
SCRIPTS=['run_lightgbm.py','run_svr_pricelags.py','run_lstm_residual_delta.py','run_transformer_residual_delta.py','run_patchtst_residual_delta.py']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(TEMPR_DATA_DIR=str(a.data_dir.resolve()),TEMPR_OUTPUT_DIR=str(out),PYTHONIOENCODING='utf-8',MPLBACKEND='Agg',OMP_NUM_THREADS=str(a.threads),MKL_NUM_THREADS=str(a.threads),TEMPR_THREADS=str(a.threads))
    records=[]
    for model in a.models:
        script=Path(__file__).resolve().parent/'code'/SCRIPTS[MODELS.index(model)]
        log=out/(model+'.log');tic=time.monotonic();print('Starting',model,flush=True)
        cmd='import os,runpy,torch; torch.set_num_threads(int(os.environ["TEMPR_THREADS"])); runpy.run_path('+repr(str(script))+',run_name="__main__")'
        # Supply the model directory to imports used by original research scripts.
        childenv=env.copy();childenv['PYTHONPATH']=str(script.parent)+os.pathsep+env.get('PYTHONPATH','')
        with log.open('w',encoding='utf8') as f:r=subprocess.run([sys.executable,'-u','-c',cmd],env=childenv,stdout=f,stderr=subprocess.STDOUT)
        records.append({'model':model,'exit_code':r.returncode,'seconds':time.monotonic()-tic,'log':log.name})
        (out/'corrected_run.json').write_text(json.dumps({'split_protocol':'target-end-purged-v3','python':sys.version,'platform':platform.platform(),'threads':a.threads,'runs':records},indent=2),encoding='utf8')
        print(records[-1],flush=True)
        if r.returncode:raise SystemExit(f'{model} failed; inspect {log}')

if __name__=='__main__':main()
