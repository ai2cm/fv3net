import logging
from dataclasses import dataclass

import gcsfs
import numpy as np
import xarray as xr

from fv3net.regression import reshape

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler("dataset_handler.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)


SAMPLE_DIMS = ["initialization_time", "grid_yt", "grid_xt", "tile"]


@dataclass
class BatchGenerator:
    gcs_data_dir: str
    files_per_batch: int
    train_frac: float
    test_frac: float
    num_train_batches: int = None
    gcs_project: str = "vcm-ml"
    random_seed: int = 1234

    def __post_init__(self):
        """Randomly splits the list of zarrs in the gcs_data_dir into train/test

        Returns:

        """
        self.fs = gcsfs.GCSFileSystem(project=self.gcs_project)
        zarr_urls = self.fs.ls(self.gcs_data_dir)
        self.train_file_batches, self.test_file_batches = self._split_train_test_files(
            zarr_urls
        )

    def generate_batches(self, batch_type="train"):
        """

        Args:
            batch_type: train or test

        Returns:
            dataset of vertical columns shuffled within each training batch
        """
        if batch_type == "train":
            grouped_urls = self.train_file_batches
        elif batch_type == "test":
            grouped_urls = self.test_file_batches
        for file_batch_urls in grouped_urls:
            fs_paths = [self.fs.get_mapper(url) for url in file_batch_urls]
            ds = xr.concat(map(xr.open_zarr, fs_paths), "initialization_time")
            ds_shuffled = self._reshape_and_shuffle(ds)
            yield ds_shuffled

    def _split_train_test_files(self, zarr_urls):
        """

        Args:
            zarr_urls: list of zarr files on GCS

        Returns:
            train and test arrays, randomly split by fraction train_frac/test_frac
        """
        num_total_files = len(zarr_urls)
        num_train_files = int(num_total_files * self.train_frac)
        num_test_files = num_total_files - num_train_files
        if not self.num_train_batches:
            self.num_train_batches = int(
                (num_train_files / self.files_per_batch)
                + min(1, num_train_files % self.files_per_batch)
            )
        num_test_batches = int(num_test_files / self.files_per_batch)

        np.random.seed(self.random_seed)
        np.random.shuffle(zarr_urls)

        train_file_batches = [
            zarr_urls[
                batch_num
                * self.files_per_batch : (batch_num + 1)
                * self.files_per_batch
            ]
            for batch_num in range(self.num_train_batches - 1)
        ]
        test_file_batches = [
            zarr_urls[
                self.num_train_batches
                + batch_num * self.files_per_batch : self.num_train_batches
                + (batch_num + 1) * self.files_per_batch
            ]
            for batch_num in range(num_test_batches - 1)
        ]
        return train_file_batches, test_file_batches

    def _reshape_and_shuffle(
        self, ds,
    ):
        """

        Args:
            ds: xarray dataset of feature and target variables with
            time/spatial dimensions

        Returns:
            xarray dataset with dimensions (except for vertical dim) stacked into a
            single sample dimension, randomly shuffled
        """
        ds_stacked = (
            ds.stack(sample=SAMPLE_DIMS)
            .transpose("sample", "pfull")
            .reset_index("sample")
        )
        ds_shuffled = reshape.shuffled(ds_stacked, "sample", self.random_seed)
        return ds_shuffled
