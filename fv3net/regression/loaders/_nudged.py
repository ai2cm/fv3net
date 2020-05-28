import os
import logging
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
from ._sequences import FunctionOutputSequence

logger = logging.getLogger(__name__)

TIMESCALE_OUTDIR_TEMPLATE = "outdir-*h"
SIMULATION_TIMESTEPS_PER_HOUR = 4
NUDGING_FILES = [
    "before_dynamics",
    "after_dynamics",
    "after_physics",
    "nudging_tendencies",
]

SAMPLE_DIM = "sample"
NUDGED_TIME_DIM = "time"
Z_DIM = "z"

BatchSequence = Sequence[xr.Dataset]
TimestepMappeer = Mapping[str, xr.Dataset]


def load_nudging_batches(
    url: str,
    data_vars: Iterable[str],
    nudging_timescale_hr: Union[int, float] = 3,
    num_times_in_batch: int = 10,
    num_batches: int = None,
    random_seed: int = 0,
    mask_to_surface_type: str = None,
    rename_variables: Mapping[str, str] = None,
    initial_time_skip_hr: int = 0,
    n_times: int = None,
):
    """temporary loader while transforms being developed"""

    mapper = open_nudged_mapper(
        url,
        nudging_timescale_hr,
        initial_time_skip_hr=initial_time_skip_hr,
        n_times=n_times,
    )
    tstep_mapper = mapper.merge_sources(["after_physics", "nudging_tendencies"])

    return _load_nudging_batch(
        tstep_mapper,
        data_vars,
        num_times_in_batch=num_times_in_batch,
        num_batches=num_batches,
        random_seed=random_seed,
        mask_to_surface_type=mask_to_surface_type,
        rename_variables=rename_variables,
        initial_time_skip_hr=initial_time_skip_hr,
        n_times=n_times,
    )


def _load_nudging_batches(
    nudged_tstep_mapper: TimestepMappeer,
    data_vars: Iterable[str],
    num_times_in_batch: int = 10,
    num_batches: int = None,
    random_seed: int = 0,
    mask_to_surface_type: str = None,
    rename_variables: Mapping[str, str] = None,
    initial_time_skip_hr: int = 0,
    n_times: int = None,
) -> Union[Sequence[BatchSequence], BatchSequence]:
    """
    Get a sequence of batches from a nudged-run zarr store.

    Args:
        nudged_tstep_mapper: Mapping object from fv3 time strings to nudged fv3
            output data
        data_vars: Variable names used to subset batched datasets
        num_times_in_batch (optional): number of times to include
            in a single batch item.  Overridden by num_batches
        num_batches (optional): number of batches to split the
            input samples into.  Overrides num_samples_in_batch.
        random_seed (optional): A seed for the RNG state used in shuffling operations
        mask_to_surface_type (optional): Flag selector for surface type masking.
            Requires "land_sea_mask" exists in the loaded dataset.  Note: as currently
            implemented NaN drop may reduce the batch size under requested
            number of samples.
        rename_variables (optional): A mapping to update any variable names in the
            dataset prior to the selection of input/output variables
        initial_time_skip_hr (optional): Length of model inititialization (in hours)
            to omit from the batching operation
        n_times (optional): Number of times (by index) to include in the
            batch resampling operation
    """

    random = np.random.RandomState(random_seed)
    timesteps = nudged_tstep_mapper.keys()
    random.shuffle(timesteps)

    # TODO: Overlaps with _validated_num_batches in _one_step
    if num_batches is None:
        num_batches = len(timesteps) // num_times_in_batch
    elif num_batches * num_times_in_batch > len(timesteps):
        raise ValueError(
            f"Not enough timesteps (n={len(timesteps)} to create requested "
            f"{num_batches} batches of size {num_times_in_batch}"
        )

    logger.info(f"Creating {num_batches} batches of size {num_times_in_batch}")
    batched_timesteps = []
    for i in range(num_batches):
        start = i * num_times_in_batch
        end = start + num_times_in_batch
        batched_timesteps.append(timesteps[start:end])

    batch_loader = partial(
        _load_nudging_batch,
        nudged_tstep_mapper,
        data_vars,
        rename_variables,
        mask_to_surface_type,
    )
    return FunctionOutputSequence(batch_loader, batched_timesteps)


# TODO: changed toplevel constants to shared constants


def _load_nudging_batch(
    timestep_mapper, data_vars, rename_variables, mask_to_surface_type, tstep_keys
) -> xr.Dataset:

    logger.debug(f"Loading batch with mapper keys: {tstep_keys}")
    batch = [timestep_mapper[timestep] for timestep in tstep_keys]
    batch_ds = xr.concat(batch, NUDGED_TIME_DIM)
    batch_ds = batch_ds.rename(rename_variables)
    if mask_to_surface_type is not None:
        batch_ds = vcm.mask_to_surface_type(batch_ds, mask_to_surface_type)
    batch_ds = safe.get_variables(batch_ds, data_vars)

    # Into memory we go
    batch_ds = batch_ds.load()

    stack_dims = [dim for dim in batch_ds.dims if dim != Z_DIM]
    stacked_batch_ds = safe.stack_once(
        batch_ds, SAMPLE_DIM, stack_dims, allowed_broadcast_dims=[Z_DIM]
    )
    stacked_batch_ds = stacked_batch_ds.dropna(SAMPLE_DIM)

    if len(stacked_batch_ds[SAMPLE_DIM]) == 0:
        raise ValueError(
            "No Valid samples detected. Check for errors in the training data."
        )

    return stacked_batch_ds


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

    def merge_sources(self, source_names: Iterable[str]) -> NudgedTimestepMapper:
        """
        Combine nudging data sources into single dataset
        """

        combined_ds = xr.Dataset()
        for source in source_names:
            ds = self._nudged_ds[source]
            self._check_dvar_overlap(combined_ds, ds)

            combined_ds = combined_ds.merge(ds)

        return NudgedTimestepMapper(combined_ds)

    # TODO: group by operations
    #   groupby source is easy since we could just return the self._nudge_ds mapping
    #   groupby time needs merged ds for all timesteps

    @staticmethod
    def _check_dvar_overlap(ds1, ds2):

        ds1_vars = set(ds1.datavars.keys())
        ds2_vars = set(ds2.data_vars.keys())
        overlap = ds1_vars & ds2_vars
        if overlap:
            raise ValueError(
                "Could not combine requested nudged data sources due to "
                f"overlapping variables {overlap}"
            )


def open_nudged_mapper(
    url: str,
    nudging_timescale_hr: Union[int, float],
    initial_time_skip_hr: int = 0,
    n_times: int = None,
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
