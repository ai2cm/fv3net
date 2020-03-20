import backoff
import logging
from dataclasses import dataclass
from typing import List
import numpy as np
import xarray as xr
import vcm
from vcm.cloud.fsspec import get_fs
from vcm.cubedsphere.constants import COORD_Z_CENTER, INIT_TIME_DIM

SAMPLE_DIM = "sample"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler("dataset_handler.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)


@dataclass
class BatchGenerator:
    data_vars: List[str]
    gcs_data_dir: str
    files_per_batch: int
    num_batches: int = None
    random_seed: int = 1234
    mask_to_surface_type: str = "none"

    def __post_init__(self):
        """ Group the input zarrs into batches for training

        Returns:
            nested list of zarr paths, where inner lists are the sets of zarrs used
            to train each batch
        """
        self.fs = get_fs(self.gcs_data_dir)
        logger.info(f"Reading data from {self.gcs_data_dir}.")
        zarr_urls = [
            zarr_file
            for zarr_file in self.fs.ls(self.gcs_data_dir)
            if "grid_spec" not in zarr_file
        ]
        total_num_input_files = len(zarr_urls)
        logger.info(f"Number of .zarrs in GCS train data dir: {total_num_input_files}.")
        np.random.seed(self.random_seed)
        np.random.shuffle(zarr_urls)
        self.num_batches = self._validated_num_batches(total_num_input_files)
        logger.info(f"{self.num_batches} data batches generated for model training.")
        self.train_file_batches = [
            zarr_urls[
                batch_num
                * self.files_per_batch : (batch_num + 1)
                * self.files_per_batch
            ]
            for batch_num in range(self.num_batches)
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
            try:
                ds_shuffled = self._create_training_batch_with_retries(file_batch_urls)
            except ValueError:
                logger.error(
                    f"Failed to generate batch from files {file_batch_urls}."
                    "Skipping to next batch."
                )
                continue
            yield ds_shuffled

    @backoff.on_exception(backoff.expo, RuntimeError, max_tries=3)
    def _create_training_batch_with_retries(self, urls):
        timestep_paths = [self.fs.get_mapper(url) for url in urls]
        try:
            ds = xr.concat(map(xr.open_zarr, timestep_paths), INIT_TIME_DIM)
            ds = vcm.mask_to_surface_type(ds, self.mask_to_surface_type)
            ds_stacked = stack_and_drop_nan_samples(ds).unify_chunks()
            ds_shuffled = _shuffled(ds_stacked, SAMPLE_DIM, self.random_seed)
            return ds_shuffled
        except ValueError as e:
            # error when attempting to read from GCS that sometimes resolves on retry
            if "array not found at path" in str(e):
                logger.error(
                    f"Error reading data from {timestep_paths}, will retry. {e}"
                )
                raise RuntimeError(str(e))
            # other errors that will not recover on retry
            else:
                logger.error(f"Error reading data from {timestep_paths}. {e}")
                raise e

    def _validated_num_batches(self, total_num_input_files):
        """ check that the number of batches (if provided) and the number of
        files per batch are reasonable given the number of zarrs in the input data dir.
        If their product is greater than the number of input files, number of batches
        is changed so that (num_batches * files_per_batch) < total files.

        Returns:
            Number of batches to use for training
        """
        if not self.num_batches:
            num_train_batches = total_num_input_files // self.files_per_batch
        elif self.num_batches * self.files_per_batch > total_num_input_files:
            if self.num_batches > total_num_input_files:
                raise ValueError("Fewer input files than number of requested batches.")
            num_train_batches = total_num_input_files // self.files_per_batch
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


def stack_and_drop_nan_samples(ds):
    """

    Args:
        ds: xarray dataset

    Returns:
        xr dataset stacked into sample dimension and with NaN elements dropped
         (the masked out land/sea type)
    """
    ds = (
        ds.stack({SAMPLE_DIM: [dim for dim in ds.dims if dim != COORD_Z_CENTER]})
        .transpose(SAMPLE_DIM, COORD_Z_CENTER)
        .dropna(SAMPLE_DIM)
    )
    return ds
