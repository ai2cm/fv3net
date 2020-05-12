import collections
import backoff
import functools
import logging
from typing import Iterable, List, Sequence, Callable, Mapping
import numpy as np
import xarray as xr

import vcm
from vcm import safe
from vcm.cloud.fsspec import get_fs

__all__ = ["load_one_step_batches"]

SAMPLE_DIM = "sample"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler("dataset_handler.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)


class BatchSequence(collections.abc.Sequence):
    def __init__(self, loader_function: Callable, args_sequence: Sequence[Iterable]):
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


def get_time_list(url_list_sequence: Sequence[List[str]]):
    """Given a sequence of lists of URLs, return a list of times present in those
    lists.
    """
    time_list = []
    for url_list in url_list_sequence:
        time_list.extend(map(_url_to_datetime, url_list))
    return time_list


def load_one_step_batches(
    data_path: str,
    input_variables: Iterable[str],
    output_variables: Iterable[str],
    files_per_batch: int,
    num_batches: int = None,
    random_seed: int = 1234,
    mask_to_surface_type: str = None,
    init_time_dim_name: str = "initial_time",
    z_dim_name: str = "z",
    rename_variables: Mapping[str, str] = None,
) -> Sequence:
    """Get a sequence of batches from one-step zarr stores.

    Args:
        data_path: location of directory containing zarr stores
        input_variables: names of inputs
        output_variables: names of outputs
        files_per_batch: number of zarr stores used to create each batch
        num_batches (optional): number of batches to create. By default, use all the
            available trianing data.
        random_seed (optional): seed value for random number generator
        mask_to_surface_type: mask data points to ony include the indicated surface type
        init_time_dim_name: name of the initialization time dimension
        z_dim_name: name of the vertical dimension
        rename_variables: mapping of variables to rename,
            from data names to standard names
    """
    if rename_variables is None:
        rename_variables = {}
    data_vars = list(input_variables) + list(output_variables)
    fs = get_fs(data_path)
    logger.info(f"Reading data from {data_path}.")
    zarr_urls = [
        zarr_file for zarr_file in fs.ls(data_path) if "grid_spec" not in zarr_file
    ]
    logger.info(f"Number of .zarrs in GCS train data dir: {len(zarr_urls)}.")
    random = np.random.RandomState(random_seed)
    random.shuffle(zarr_urls)
    num_batches = _validated_num_batches(len(zarr_urls), files_per_batch, num_batches)
    logger.info(f"{num_batches} data batches generated for model training.")
    url_list_sequence = list(
        zarr_urls[batch_num * files_per_batch : (batch_num + 1) * files_per_batch]
        for batch_num in range(num_batches)
    )
    load_batch = functools.partial(
        _load_one_step_batch,
        fs,
        data_vars,
        rename_variables,
        init_time_dim_name,
        z_dim_name,
        mask_to_surface_type,
        random,
    )
    args_sequence = [(item,) for item in url_list_sequence]
    return (
        BatchSequence(load_batch, args_sequence),
        get_time_list(url_list_sequence),
    )


def _load_one_step_batch(
    fs,
    data_vars: Iterable[str],
    rename_variables: Mapping[str, str],
    init_time_dim_name: str,
    z_dim_name: str,
    mask_to_surface_type: str,
    random,
    url_list: Iterable[str],
):
    # TODO refactor this I/O. since this logic below it is currently
    # impossible to test.
    data = _load_datasets(fs, url_list)
    ds = xr.concat(data, init_time_dim_name)
    ds = ds.rename(rename_variables)
    ds = safe.get_variables(ds, data_vars)
    if mask_to_surface_type is not None:
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
    ds = ds_no_nan.load()
    return _shuffled(ds, SAMPLE_DIM, random)


@backoff.on_exception(backoff.expo, (ValueError, RuntimeError), max_tries=3)
def _load_datasets(fs, urls):
    return_list = []
    for url in urls:
        mapper = fs.get_mapper(url)
        ds = xr.open_zarr(mapper)
        return_list.append(ds)
    return return_list


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


def _shuffled(dataset, dim, random):
    chunks_default = (len(dataset[dim]),)
    chunks = dataset.chunks.get(dim, chunks_default)
    indices = _chunk_indices(chunks)
    shuffled_inds = _shuffled_within_chunks(indices, random)
    return dataset.isel({dim: shuffled_inds})


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
