import argparse
from dataclasses import dataclass
import numpy as np
import xarray as xr

import time

@dataclass
class BatchGenerator:
    ds: xr.Dataset
    batch_size: int
    train_frac: float
    test_frac: float
    num_batches: int = None

    def __post_init__(self):
        self.num_total_samples = len(self.ds.sample)
        self.num_train_samples = int(self.num_total_samples * self.train_frac)
        if not self.num_batches:
            self.num_batches = int(
                (self.num_train_samples / self.batch_size)
                + min(1, self.num_train_samples % self.batch_size))

    def generate_train_batches(self):
        for i in range(self.num_batches):
            ds_train_batch = self.ds.isel(
                sample=slice(i * self.batch_size, (i+1) * self.batch_size))
            yield ds_train_batch

    def generate_test_set(self):
        return self.ds.isel(sample=slice(self.num_train_samples, None))

