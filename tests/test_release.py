import unittest,json,ast,hashlib
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(R))
from infer_figures import load_inputs,make_task
class ReleaseTests(unittest.TestCase):
 def test_chronology_and_label_separation(self):
  c,h,x=load_inputs();obs=pd.read_csv(R/'data/figures/observations.csv',parse_dates=['time'])
  self.assertTrue(x.time.equals(obs.time));self.assertEqual(len(obs),672)
  task=make_task(h,x.iloc[:48],c['features'])
  self.assertEqual(len(task['target']),1024);self.assertNotIn('price_day_ahead',task['future_covariates'])
  np.testing.assert_array_equal(task['target'],h.price_day_ahead.to_numpy(np.float32))
 def test_no_saved_lora_input(self):
  for f in (R/'data/figures').glob('*.csv'):
   self.assertFalse(any('lora' in k.lower() for k in pd.read_csv(f,nrows=0).columns))
  self.assertFalse((R/'data/demo').exists())
 def test_original_adapter_hash(self):
  c=json.loads((R/'config/figure_demo.json').read_text(encoding='utf-8'));self.assertEqual(hashlib.sha256((R/'models/chronos2_lora/adapter_model.safetensors').read_bytes()).hexdigest(),c['adapter_sha256'])
 def test_code_parses(self):
  for folder in ['src','scripts','forecast/code']:
   for f in (R/folder).rglob('*.py'):ast.parse(f.read_text(encoding='utf-8-sig'))
 def test_public_boundary(self):
  for rel in ['forecast/data','forecast/outputs','results','artwork','data/figure_sources','data/fig1_sources','data/reference_tables']:
   self.assertFalse((R/rel).exists(),rel)
  self.assertEqual(len(list((R/'data/figures').glob('*.csv'))),4)
if __name__=='__main__':unittest.main()
