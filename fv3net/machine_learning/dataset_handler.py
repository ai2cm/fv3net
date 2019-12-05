import argparse
from dataclasses import dataclass
import numpy as np
import random
import xarray as xr


@dataclass
class BatchGenerator:
    ds: xr.Dataset
    batch_size: int
    train_frac: float
    test_frac: float
    num_batches: int = None

    def __post_init__(self):
        if not self.num_batches:
            self.num_batches = (len(self.ds.sample) / self.batch_size) \
                               + min(1, len(self.ds.sample) % self.batch_size)

    def __iter__(self):
        for i in range(self.batches):
            ds_train_batch = self.ds.isel(
                sample=slice(i * self.batch_size, (i+1) * self.batch_size))
            ds_train_batch, ds_test_batch = train_test_split(ds_train_batch)
            yield ds_train_batch, ds_test_batch


def train_test_split(
        ds,
        train_frac,
        test_frac,
        num_samples=None,
        random_seed=1234
):
    if train_frac + test_frac > 1.:
        raise ValueError("Train + test fraction sums must be <= 1.")
    sample_indices = ds.sample.values
    total_num_datapoints = len(sample_indices)
    if not num_samples:
        num_samples = total_num_datapoints
    num_train = int(train_frac * num_samples)
    num_test = int(test_frac * num_samples)
    random.seed(random_seed)
    random.shuffle(sample_indices)
    ds_train = ds.isel(sample=sample_indices[:num_train])
    ds_test = ds.isel(sample=sample_indices[-num_test:])
    return ds_train, ds_test

