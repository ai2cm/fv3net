import collections
import warnings
import backoff
import functools
import logging
from dataclasses import dataclass
from typing import List, Iterable
import numpy as np
import xarray as xr

import vcm
from vcm import safe
from vcm.cloud.fsspec import get_fs

SAMPLE_DIM = "sample"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler("dataset_handler.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)


class BatchSequence(collections.abc.Sequence):
    def __init__(self, loader_function, args_sequence):
        """Create a sequence of Batch objects from a function which produces a single
        batch, and a sequence of arguments to that function.
        """
        self._loader = loader_function
        self._args = args_sequence

    def __getitem__(self, item):
        return self._loader(*self._args[item])

    def __len__(self):
        return len(self._args)


def _url_to_datetime(url):
    return vcm.cast_to_datetime(
        vcm.parse_datetime_from_str(vcm.parse_timestep_str_from_path(url))
    )


def get_time_list(url_list_sequence):
    time_list = []
    for url_list in url_list_sequence:
        time_list.extend(map(_url_to_datetime, url_list))
    return time_list


def load_one_step_batches(
    input_variables: Iterable[str],
    output_variables: Iterable[str],
    gcs_data_dir: str,
    files_per_batch: int,
    num_batches: int = None,
    random_seed: int = 1234,
    mask_to_surface_type: str = "none",
    init_time_dim_name="initial_time",
    z_dim_name="z",
    rename_variables=None,
):
    """Get a sequence of batches from one-step output.

    Args:
        data_vars: variable names to load
        gcs_data_dir: gcs location of data
        files_per_batch: number of files used to create each batch
        num_batches (optional): number of batches to create. By default, use all the
            available trianing data.
    """
    if rename_variables is None:
        rename_variables = {}
    data_vars = list(input_variables) + list(output_variables)
    fs = get_fs(gcs_data_dir)
    logger.info(f"Reading data from {gcs_data_dir}.")
    zarr_urls = [
        zarr_file for zarr_file in fs.ls(gcs_data_dir) if "grid_spec" not in zarr_file
    ]
    logger.info(f"Number of .zarrs in GCS train data dir: {len(zarr_urls)}.")
    random = np.random.RandomState(random_seed)
    random.shuffle(zarr_urls)
    num_batches = _validated_num_batches(len(zarr_urls), files_per_batch, num_batches)
    logger.info(f"{num_batches} data batches generated for model training.")
    url_list_sequence = (
        (zarr_urls[batch_num * files_per_batch : (batch_num + 1) * files_per_batch],)
        for batch_num in range(num_batches)
    )
    load_batch = functools.partial(
        _load_one_step_batch,
        data_vars,
        rename_variables,
        init_time_dim_name,
        z_dim_name,
        mask_to_surface_type,
        random,
    )
    return (
        BatchSequence(load_batch, url_list_sequence),
        get_time_list(url_list_sequence),
    )


def _load_one_step_batch(
    data_vars,
    rename_variables,
    init_time_dim_name,
    z_dim_name,
    mask_to_surface_type,
    random,
    url_list,
):
    # TODO refactor this I/O. since this logic below it is currently
    # impossible to test.
    data = _load_datasets(url_list)
    ds = xr.concat(data, init_time_dim_name)
    ds = ds.rename(rename_variables)
    ds = safe.get_variables(ds, data_vars)
    ds = vcm.mask_to_surface_type(ds, mask_to_surface_type)
    stack_dims = [dim for dim in ds.dims if dim != z_dim_name]
    ds_stacked = safe.stack_once(
        ds,
        SAMPLE_DIM,
        stack_dims,
        allowed_broadcast_dims=[z_dim_name, init_time_dim_name],
    )

    ds_no_nan = ds_stacked.dropna(SAMPLE_DIM)

    if len(ds_no_nan[SAMPLE_DIM]) == 0:
        raise ValueError(
            "No Valid samples detected. Check for errors in the training data."
        )

    return _shuffled(ds_no_nan, SAMPLE_DIM, random)


@backoff.on_exception(backoff.expo, (ValueError, RuntimeError), max_tries=3)
def _load_datasets(self, urls):
    timestep_paths = [self.fs.get_mapper(url) for url in urls]
    return [xr.open_zarr(path).load() for path in timestep_paths]


def _validated_num_batches(total_num_input_files, files_per_batch, num_batches=None):
    """ check that the number of batches (if provided) and the number of
    files per batch are reasonable given the number of zarrs in the input data dir.

    Returns:
        Number of batches to use for training
    """
    if num_batches is None:
        num_train_batches = total_num_input_files // files_per_batch
    elif num_batches * files_per_batch > total_num_input_files:
        raise ValueError(
            f"Number of input_files {total_num_input_files} "
            f"cannot create {num_batches} batches of size {files_per_batch}."
        )
    else:
        num_train_batches = num_batches
    return num_train_batches


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
        warnings.warn(
            "BatchGenerator has been deprecated and will be removed, "
            "use a function which returns a BatchSequence instead",
            warnings.DeprecationWarning,
        )
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

    def generate_batches(self, coord_z_center, init_time_dim):
        """

        Args:
            batch_type: train or test

        Returns:
            dataset of vertical columns shuffled within each training batch
        """
        grouped_urls = self.train_file_batches
        for file_batch_urls in grouped_urls:
            yield self._create_training_batch(
                file_batch_urls, coord_z_center, init_time_dim
            )

    @backoff.on_exception(backoff.expo, (ValueError, RuntimeError), max_tries=3)
    def _load_datasets(self, urls):
        timestep_paths = [self.fs.get_mapper(url) for url in urls]
        return [xr.open_zarr(path).load() for path in timestep_paths]

    def _create_training_batch(self, urls, coord_z_center, init_time_dim):
        # TODO refactor this I/O. since this logic below it is currently
        # impossible to test.
        data = self._load_datasets(urls)
        ds = xr.concat(data, init_time_dim)
        ds = safe.get_variables(ds, self.data_vars)
        ds = vcm.mask_to_surface_type(ds, self.mask_to_surface_type)
        stack_dims = [dim for dim in ds.dims if dim != coord_z_center]
        ds_stacked = safe.stack_once(
            ds,
            SAMPLE_DIM,
            stack_dims,
            allowed_broadcast_dims=[coord_z_center, init_time_dim],
        )

        ds_no_nan = ds_stacked.dropna(SAMPLE_DIM)

        if len(ds_no_nan[SAMPLE_DIM]) == 0:
            raise ValueError(
                "No Valid samples detected. Check for errors in the training data."
            )

        return _shuffled(ds_no_nan, SAMPLE_DIM, self.random_seed)

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
                raise ValueError(
                    f"Number of input_files {total_num_input_files} "
                    f"smaller than {self.num_batches}."
                )
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


def _shuffled_within_chunks(indices, random):
    # We should only need to set the random seed once (not every time)
    return np.concatenate([random.permutation(index) for index in indices])


def _shuffled(dataset, dim, random):
    chunks_default = (len(dataset[dim]),)
    chunks = dataset.chunks.get(dim, chunks_default)
    indices = _chunk_indices(chunks)
    shuffled_inds = _shuffled_within_chunks(indices, random)
    return dataset.isel({dim: shuffled_inds})


def stack_and_drop_nan_samples(ds, coord_z_center):
    """

    Args:
        ds: xarray dataset

    Returns:
        xr dataset stacked into sample dimension and with NaN elements dropped
         (the masked out land/sea type)
    """
    # TODO delete this function
    ds = (
        safe.stack_once(
            ds, SAMPLE_DIM, [dim for dim in ds.dims if dim != coord_z_center]
        )
        .transpose(SAMPLE_DIM, coord_z_center)
        .dropna(SAMPLE_DIM)
    )
    return ds
