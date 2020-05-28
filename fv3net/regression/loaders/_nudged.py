import os
import fsspec
import xarray as xr
import pandas as pd
import numpy as np
import zarr.storage as zstore
from typing import Sequence, Iterable, Mapping, Union
from functools import partial
from itertools import product
from pathlib import Path

import vcm
from vcm import cloud, safe
from vcm.cubedsphere.constants import TIME_FMT
from . import _transform as transform
from ._sequences import FunctionOutputSequence

TIMESCALE_OUTDIR_TEMPLATE = "outdir-*h"
SIMULATION_TIMESTEPS_PER_HOUR = 4
NUDGING_FILES = [
    "before_dynamics",
    "after_dynamics",
    "after_physics",
    "nudging_tendencies"
]

SAMPLE_DIM = "sample"
NUDGED_TIME_DIM = "time"

BatchSequence = Sequence[xr.Dataset]


def load_nudging_batches(
    data_path: str,
    *variable_names: Iterable[str],
    timescale_hours: Union[int, float] = 3,
    num_samples_in_batch: int = 13824,
    num_batches: int = None,
    random_seed: int = 0,
    mask_to_surface_type: str = None,
    z_dim_name: str = "z",
    rename_variables: Mapping[str, str] = None,
    time_dim_name: str = "time",
    initial_time_skip_hr: int = 0,
    n_times: int = None,
) -> Union[Sequence[BatchSequence], BatchSequence]:
    """
    Get a sequence of batches from a nudged-run zarr store.

    Args:
        data_path: location of directory containing zarr stores
        *variable_names: any number of sequences of variable names. One Sequence will be
            returned for each of the given sequences. The "sample" dimension will be
            identical across each of these sequences.
        timescale_hours (optional): timescale of the nudging for the simulation
            being used as input.
        num_samples_in_batch (optional): number of samples to include
            in a single batch item.  Overridden by num_batches
        num_batches (optional): number of batches to split the
            input samples into.  Overrides num_samples_in_batch.
        random_seed (optional): A seed for the RNG state used in shuffling operations
        mask_to_surface_type (optional): Flag selector for surface type masking.
            Requires "land_sea_mask" exists in the loaded dataset.  Note: as currently
            implemented NaN drop may reduce the batch size under requested
            number of samples.
        z_dim_name (optional): vertical dimension name to retain in the dimension
            stacking procedure
        rename_variables (optional): A mapping to update any variable names in the
            dataset prior to the selection of input/output variables
        time_dim_name (optional): Time dimension name to use for selection from input
            data
        initial_time_skip_hr (optional): Length of model inititialization (in hours)
            to omit from the batching operation
        n_times (optional): Number of times (by index) to include in the
            batch resampling operation
    """

    nudged_output_path = _get_path_for_nudging_timescale(
        nudged_output_dirs, timescale_hours
    )

    datasets_to_batch = _load_datasets(
        fs, nudged_output_path, variable_names, rename_variables
    )

    batched_sequences = []
    for dataset in datasets_to_batch:
        start = initial_time_skip_hr * SIMULATION_TIMESTEPS_PER_HOUR
        end = start + n_times
        dataset = dataset.isel({time_dim_name: slice(start, end)})

        if mask_to_surface_type is not None:
            dataset = vcm.mask_to_surface_type(dataset, mask_to_surface_type)

        stack_dims = [dim for dim in dataset.dims if dim != z_dim_name]
        ds_stacked = safe.stack_once(
            dataset, SAMPLE_DIM, stack_dims, allowed_broadcast_dims=[z_dim_name]
        )

        total_samples = ds_stacked.sizes[SAMPLE_DIM]

        sample_slice_sequence = _get_batch_slices(
            total_samples, num_samples_in_batch, num_batches=num_batches
        )
        random = np.random.RandomState(random_seed)
        random.shuffle(sample_slice_sequence)

        loader_func = partial(_load_nudging_batch, ds_stacked, random)

        batched_sequences.append(
            FunctionOutputSequence(loader_func, sample_slice_sequence)
        )

    if len(batched_sequences) > 1:
        return batched_sequences
    else:
        return batched_sequences[0]


def _load_nudging_batch(
    stacked_ds: xr.Dataset, random: np.random.RandomState, batch_slice: slice
) -> xr.Dataset:

    batch = stacked_ds.isel({SAMPLE_DIM: batch_slice})
    batch = batch.load()
    batch_no_nan = batch.dropna(SAMPLE_DIM)

    if len(batch_no_nan[SAMPLE_DIM]) == 0:
        raise ValueError(
            "No Valid samples detected. Check for errors in the training data."
        )

    return transform.shuffled(batch_no_nan, SAMPLE_DIM, random)


def _get_batch_slices(
    num_samples: int, samples_per_batch: int, num_batches: int = None
):

    if num_batches is not None:
        batch_size = num_samples // num_batches
    else:
        batch_size = samples_per_batch
        num_batches = num_samples // samples_per_batch

    if batch_size == 0 or num_batches == 0:
        raise ValueError(
            "The number of input samples is insufficient to create a batch for "
            f"requested samples_per_batch={samples_per_batch} or "
            f"requested num_batches={num_batches}."
        )

    slices = []
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        slices.append(slice(start, end))

    return slices


def _get_path_for_nudging_timescale(nudged_output_dirs, timescale_hours, tol=1e-5):
    """
    Timescales are allowed to be floats which makes finding correct output
    directory a bit trickier.  Currently checking by looking for difference
    between parsed timescale from folder name and requested timescale that
    is approximately zero (with requested tolerance).

    Built on assumed outdir-{timescale}h format
    """

    for dirpath in nudged_output_dirs:
        dirname = Path(dirpath).name
        avail_timescale = float(dirname.split("-")[-1].strip("h"))
        if abs(timescale_hours - avail_timescale) < tol:
            return dirpath
    else:
        raise KeyError(
            "Could not find nudged output directory appropriate for timescale: "
            "{timescale_hours}"
        )


class BaseMapper:

    def __init__(self, *args):
        raise NotImplementedError("Don't use the base class!")

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, key: str) -> xr.Dataset:
        raise NotImplementedError()

    def keys(self):
        raise NotImplementedError()


class NudgedTimestepMapper(BaseMapper):

    def __init__(self, ds, time_dim_name=NUDGED_TIME_DIM):
        self.ds = ds
        self.time_dim = time_dim_name

    def __getitem__(self, key: str) -> xr.Dataset:
        dt64 = np.datetime64(vcm.parse_datetime_from_str(key))
        return self.ds.sel({self.time_dim: dt64})

    def keys(self):
        return [
            time.strftime(TIME_FMT)
            for time in pd.to_datetime(self.ds[self.time_dim].values)
        ]


class NudgedMapperAllSources(BaseMapper):
    """
    Get all nudged output zarr datasets.
    Accessible by, e.g., mapper[("before_dynamics", "20160801.001500")]
    """

    def __init__(self, ds_map: Mapping[str, xr.Dataset]):

        self._nudged_ds = {key: NudgedTimestepMapper(ds) for key, ds in ds_map.items()}

    def __getitem__(self, key):
        return self._nudged_ds[key[0]][key[1]]

    def keys(self):
        keys = []
        for key, mapper in self._nudged_ds.items():
            timestep_keys = mapper.keys()
            keys.extend(product((key,), timestep_keys))
        return keys


def open_nudged_mapper(
    url: str,
    nudging_timescale_hr: Union[int, float],
    initial_time_skip_hr: int = 0,
    n_times: int = None
) -> Mapping[str, xr.Dataset]:
    
    fs = cloud.get_fs(url)

    glob_url = os.path.join(url, TIMESCALE_OUTDIR_TEMPLATE)
    nudged_output_dirs = fs.glob(glob_url)

    nudged_url = _get_path_for_nudging_timescale(
        nudged_output_dirs, nudging_timescale_hr
    )
    
    datasets = {}
    for source in NUDGING_FILES:
        mapper = fs.get_mapper(os.path.join(nudged_url, f"{source}.zarr"))
        ds = xr.open_zarr(zstore.LRUStoreCache(mapper, 1024))

        start = initial_time_skip_hr * SIMULATION_TIMESTEPS_PER_HOUR
        end = None if n_times is None else start + n_times
        ds = ds.isel({NUDGED_TIME_DIM: slice(start, end)})

        datasets[source] = ds

    nudged_mapper = NudgedMapperAllSources(datasets)

    return nudged_mapper
