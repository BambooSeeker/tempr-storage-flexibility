"""Regression tests for the two forecast-evaluation boundaries audited in v25."""
import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'forecast/code'))
from chronological_split import purged_partition_indices
import shared_data as sd


class ForecastBoundaryTests(unittest.TestCase):
    def test_target_periods_are_disjoint_and_test_origins_unchanged(self):
        starts=np.datetime64('2025-01-01T00:30')+np.arange(18961)*np.timedelta64(30,'m')
        train,val,test=purged_partition_indices(starts)
        end=starts+np.timedelta64(47*30,'m')
        self.assertLess(end[train].max(),starts[val].min())
        self.assertLess(end[val].max(),starts[test].min())
        np.testing.assert_array_equal(test,np.arange(15168,18961))
        self.assertEqual((len(train),len(val),len(test)),(13604,1470,3793))

    def test_gaps_are_handled_by_time_not_fixed_row_deletion(self):
        starts=np.datetime64('2025-01-01T00:30')+np.arange(1000)*np.timedelta64(30,'m')
        starts[720:]+=np.timedelta64(2,'D');starts[800:]+=np.timedelta64(2,'D')
        train,val,test=purged_partition_indices(starts)
        self.assertEqual((len(train),len(val),len(test)),(720,80,200))

    def test_future_labels_do_not_change_training_scaler(self):
        times=np.datetime64('2025-01-01T00:30')+np.arange(1000)*np.timedelta64(30,'m')
        labels=np.arange(1000,dtype=np.float32);features=np.ones((1000,2),dtype=np.float32)
        observed=np.ones(1000,dtype=bool)
        def build(y):
            df=pd.DataFrame({'time':times,'price_day_ahead':y})
            with patch.object(sd,'load_and_clean_all_data',return_value=df),patch.object(sd,'build_features',return_value=(features,y,times,observed,['a','b'])):
                return sd.build_dataset(save_outputs=False,verbose=False)
        a=build(labels);changed=labels.copy();changed[times>=a.val_times[0]]=1e9
        b=build(changed)
        self.assertEqual((a.scaler.mean,a.scaler.std),(b.scaler.mean,b.scaler.std))
        np.testing.assert_array_equal(a.y_train_raw,b.y_train_raw)
        self.assertFalse(np.array_equal(a.y_val_raw,b.y_val_raw))

    def test_invalid_chronology_is_rejected(self):
        starts=np.array(['2025-01-01','2025-01-01','2025-01-02'],dtype='datetime64[D]')
        with self.assertRaises(ValueError):purged_partition_indices(starts)

if __name__=='__main__':unittest.main()
