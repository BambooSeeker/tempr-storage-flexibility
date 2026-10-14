"""Chronological partitions with disjoint target periods for multi-step forecasts."""
import numpy as np


def purged_partition_indices(starts, horizon=48, train_ratio=0.8, val_fraction=0.1):
    starts = np.asarray(starts, dtype='datetime64[ns]')
    if horizon < 1 or len(starts) < 3:
        raise ValueError('Positive horizon and at least three windows are required.')
    if not np.all(np.diff(starts) > np.timedelta64(0, 'ns')):
        raise ValueError('Forecast origins must be strictly increasing.')
    n = len(starts)
    test_cut = int(n * train_ratio)
    val_cut = int(test_cut * (1.0 - val_fraction))
    if not 0 < val_cut < test_cut < n:
        raise ValueError('The requested proportions produce an empty partition.')
    ends = starts + np.timedelta64(30 * (horizon - 1), 'm')
    train = np.arange(val_cut)[ends[:val_cut] < starts[val_cut]]
    val = np.arange(val_cut, test_cut)[ends[val_cut:test_cut] < starts[test_cut]]
    test = np.arange(test_cut, n)
    if not len(train) or not len(val):
        raise ValueError('Insufficient history after purging overlapping target windows.')
    assert ends[train].max() < starts[val].min()
    assert ends[val].max() < starts[test].min()
    return train, val, test
