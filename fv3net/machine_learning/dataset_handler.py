import argparse
from dataclasses import dataclass
import gcsfs
import logging
import numpy as np
import xarray as xr
import time

from fv3net.machine_learning import reshape

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler('dataset_handler.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)


SAMPLE_DIMS = ['initialization_time', 'grid_yt', 'grid_xt']

@dataclass
class BatchGenerator:
    gcs_data_dir: str
    files_per_batch: int
    train_frac: float
    test_frac: float
    num_train_batches: int = None
    gcs_project: str = 'vcm-ml'
    random_seed: int = 1234

    def __post_init__(self):
        self.fs = gcsfs.GCSFileSystem(project=self.gcs_project)
        zarr_urls = self.fs.ls(self.gcs_data_dir)
        self.train_file_batches, self.test_file_batches = \
            self._split_train_test_files(zarr_urls)

    def generate_train_batches(self):
        for file_batch_urls in self.train_file_batches:
            fs_paths = [self.fs.get_mapper(url) for url in file_batch_urls]
            ds = xr.concat(map(xr.open_zarr, fs_paths), 'initialization_time')
            ds_shuffled = self._reshape_and_shuffle(ds, SAMPLE_DIMS)
            yield ds_shuffled

    def generate_test_batches(self):
        for file_batch_urls in self.test_file_batches:
            fs_paths = [self.fs.get_mapper(url) for url in file_batch_urls]
            ds = xr.concat(map(xr.open_zarr, fs_paths), 'initialization_time')
            yield ds

    def _split_train_test_files(self, zarr_urls):
        num_total_files = len(zarr_urls)
        num_train_files = int(num_total_files * self.train_frac)
        num_test_files = num_total_files - num_train_files
        if not self.num_train_batches:
            self.num_train_batches = int(
                (num_train_files / self.files_per_batch)
                + min(1, num_train_files % self.files_per_batch))
        num_test_batches =  int(num_test_files / self.files_per_batch)

        np.random.seed(self.random_seed)
        np.random.shuffle(zarr_urls)

        train_file_batches = [
            zarr_urls[batch_num * self.files_per_batch:
                      (batch_num+1) * self.files_per_batch]
            for batch_num in range(self.num_train_batches - 1)]
        test_file_batches = [
            zarr_urls[self.num_train_batches + batch_num * self.files_per_batch:
                      self.num_train_batches + (batch_num+1) * self.files_per_batch]
            for batch_num in range(num_test_batches - 1)]
        return train_file_batches, test_file_batches

    def _reshape_and_shuffle(
            self,
            ds,
    ):
        t0 = time.time()
        ds_stacked = ds \
            .stack(sample=SAMPLE_DIMS) \
            .transpose("sample", "pfull") \
            .reset_index('sample')
        ds_shuffled = reshape.shuffled(ds_stacked, "sample", self.random_seed)
        logger.info(f"Time to shuffle and rechunk: {int(time.time()-t0)} s")
        return ds_shuffled
