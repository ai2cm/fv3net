import logging
from dataclasses import dataclass
from typing import List

import gcsfs
from math import ceil
import numpy as np
import xarray as xr

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler("dataset_handler.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)

INIT_TIME_DIM = "initialization_time"
SAMPLE_DIM = "sample"


@dataclass
class BatchGenerator:
    data_vars: List[str]
    gcs_data_dir: str
    files_per_batch: int
    num_batches: int = None
    gcs_project: str = "vcm-ml"
    random_seed: int = 1234

    def __post_init__(self):
        """ Group the input zarrs into batches for training

        Returns:
            nested list of zarr paths, where inner lists are the sets of zarrs used
            to train each batch
        """
        self.fs = gcsfs.GCSFileSystem(project=self.gcs_project)
        zarr_urls = [
            zarr_file
            for zarr_file in self.fs.ls(self.gcs_data_dir)
            if "grid_spec" not in zarr_file
        ]
        np.random.seed(self.random_seed)
        np.random.shuffle(zarr_urls)
        num_batches = self._validated_num_batches(total_num_input_files=len(zarr_urls))
        self.train_file_batches = [
            zarr_urls[
                batch_num
                * self.files_per_batch : (batch_num + 1)
                * self.files_per_batch
            ]
            for batch_num in range(num_batches - 1)
        ]

    def generate_batches(self):
        """

        Args:
            batch_type: train or test

        Returns:
            dataset of vertical columns shuffled within each training batch
        """
        grouped_urls = self.train_file_batches
        for file_batch_urls in grouped_urls:
            fs_paths = [self.fs.get_mapper(url) for url in file_batch_urls]
            ds = xr.concat(map(xr.open_zarr, fs_paths), INIT_TIME_DIM)
            ds_shuffled = _shuffled(ds, SAMPLE_DIM, self.random_seed)
            yield ds_shuffled

    @property
    def _validated_num_batches(self, total_num_input_files):
        """ check that the number of batches (if provided) and the number of
        files per batch are reasonable given the number of zarrs in the input data dir.
        If their product is greater than the number of input files, number of batches
        is changed so that (num_batches * files_per_batch) < total files.

        Returns:
            None
        """
        if not self.num_batches:
            num_train_batches = total_num_input_files // self.files_per_batch
        elif self.num_batches * self.files_per_batch > total_num_input_files:
            num_train_batches = self.num_batches - ceil(
                (self.num_batches * self.files_per_batch - total_num_input_files)
                / self.num_batches
            )
        else:
            num_train_batches = self.num_batches
        return num_train_batches


def _chunk_indices(chunks):
    indices = []

    start = 0
    for chunk in chunks:
        indices.append(list(range(start, start + chunk)))
        start += chunk
    return indices


def _shuffled_within_chunks(indices, random_seed):
    np.random.seed(random_seed)
    return np.concatenate([np.random.permutation(index) for index in indices])


def _shuffled(dataset, dim, random_seed):
    indices = _chunk_indices(dataset.chunks[dim])
    shuffled_inds = _shuffled_within_chunks(indices, random_seed)
    return dataset.isel({dim: shuffled_inds})
