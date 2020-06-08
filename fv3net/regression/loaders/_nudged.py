import os
import logging
import xarray as xr
import pandas as pd
import numpy as np
import zarr.storage as zstore
from typing import Sequence, Mapping, Union, Tuple
from itertools import product
from toolz import groupby
from pathlib import Path

import vcm
from vcm import cloud
from vcm.convenience import round_time
from vcm.cubedsphere.constants import TIME_FMT
from .constants import TIME_NAME

logger = logging.getLogger(__name__)

TIMESCALE_OUTDIR_TEMPLATE = "outdir-*h"
SIMULATION_TIMESTEPS_PER_HOUR = 4
Z_DIM_NAME = "z"


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
            f"{timescale_hours}"
        )


class GeoMapper:
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


class NudgedTimestepMapper(GeoMapper):
    """
    Basic mapper across the time dimension for any long-form
    simulation output.
    
    This mapper uses slightly different
    initialization (a dataset instead of a url) because nudge
    run information for all timesteps already exists within
    a single file, i.e., no filesystem grouping is necessary to get
    an item.
    """

    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, key: str) -> xr.Dataset:
        dt64 = np.datetime64(vcm.parse_datetime_from_str(key))
        return self.ds.sel({TIME_NAME: dt64})

    def keys(self):
        return [
            time.strftime(TIME_FMT)
            for time in pd.to_datetime(self.ds[TIME_NAME].values)
        ]


class MergeNudged(NudgedTimestepMapper):
    """
    Mapper for merging data sources available from a nudged run.
    
    Currently used to merge the nudging tendencies with the after
    physics checkpointed state information. Could be useful for
    merging prognostic run output by time in the future.
    """

    def __init__(
        self, *nudged_sources: Sequence[Union[NudgedTimestepMapper, xr.Dataset]]
    ):
        if len(nudged_sources) < 2:
            raise TypeError(
                "MergeNudged should be instantiated with two or more data sources."
            )
        nudged_sources = self._mapper_to_datasets(nudged_sources)
        self._check_dvar_overlap(*nudged_sources)
        self.ds = xr.merge(nudged_sources, join="inner")

    @staticmethod
    def _mapper_to_datasets(data_sources) -> Mapping[str, xr.Dataset]:

        datasets = []
        for source in data_sources:
            if isinstance(source, NudgedTimestepMapper):
                source = source.ds
            datasets.append(source)

        return datasets

    @staticmethod
    def _check_dvar_overlap(*ds_to_combine):
        ds_var_sets = [set(ds.data_vars.keys()) for ds in ds_to_combine]

        overlap = set()
        checked = set()
        for data_var in ds_var_sets:
            overlap |= data_var & checked
            checked |= data_var

        if overlap:
            raise ValueError(
                "Could not combine requested nudged data sources due to "
                f"overlapping variables {overlap}"
            )


class NudgedStateCheckpoints(GeoMapper):
    """
    Storage for state checkpoints from nudging runs.
    Accessible by, e.g., mapper[("before_dynamics", "20160801.001500")]
    Uses NudgedTimestepMappers for individual sources.
    """

    def __init__(self, ds_map: Mapping[str, xr.Dataset]):

        self.sources = {key: NudgedTimestepMapper(ds) for key, ds in ds_map.items()}

    def __getitem__(self, key):
        return self.sources[key[0]][key[1]]

    def keys(self):
        keys = []
        for key, mapper in self.sources.items():
            timestep_keys = mapper.keys()
            keys.extend(product((key,), timestep_keys))
        return keys


Source = str
Time = str
Checkpoint = Tuple[Source, Time]


class GroupByTime:
    # TODO: Could be generalized in if useful, nearly identical
    # function in _transform for tiles

    def __init__(self, checkpoint_data: Mapping[Checkpoint, xr.Dataset]):
        def keyfn(key):
            checkpoint, time = key
            return time

        self._checkpoint_data = checkpoint_data
        self._time_lookup = groupby(keyfn, self._checkpoint_data.keys())

    def keys(self):
        return self._time_lookup.keys()

    def __len__(self):
        return len(self.keys())

    def __getitem__(self, time: Time) -> xr.Dataset:
        checkpoints = [self._checkpoint_data[key] for key in self._time_lookup[time]]
        checkpoint_names = [key[0] for key in self._time_lookup[time]]
        ds = xr.concat(checkpoints, dim="checkpoint")
        return ds.assign_coords(checkpoint=checkpoint_names)


class SubsetTimes(GeoMapper):
    """
    Sort and subset a timestep-based mapping to skip spin-up and limit
    the number of available times.
    """

    def __init__(
        self,
        initial_time_skip_hr: int,
        n_times: Union[int, None],
        nudged_data: Mapping[str, xr.Dataset],
    ):
        timestep_keys = list(nudged_data.keys())
        timestep_keys.sort()

        start = initial_time_skip_hr * SIMULATION_TIMESTEPS_PER_HOUR
        end = None if n_times is None else start + n_times
        self._keys = timestep_keys[slice(start, end)]
        self._nudged_data = nudged_data

    def keys(self):
        return list(self._keys)

    def __getitem__(self, time: Time):
        if time not in self._keys:
            raise KeyError("Time {time} not found in SubsetTimes mapper.")
        return self._nudged_data[time]


def open_nudged(
    url: str,
    nudging_timescale_hr: Union[int, float],
    merge_files: Tuple[str] = ("after_physics", "nudging_tendencies"),
    initial_time_skip_hr: int = 0,
    n_times: int = None,
) -> Mapping[str, xr.Dataset]:
    """
    Load nudging data mapper for use with training.  Currently merges the
    two files after_physics and nudging_tendencies, which are required I/O
    for the training

    Args:
        url: Path to directory with nudging output (not including the timescale
            subdirectories, e.g., outdir-3h)
        timescale_hours: timescale of the nudging for the simulation
            being used as input.
        merge_files (optionsl): underlying nudging zarr datasets to combine
            into a MergeNudged mapper
        initial_time_skip_hr (optional): Length of model inititialization (in hours)
            to omit from the batching operation
        n_times (optional): Number of times (by index) to include in the
            batch resampling operation
    """

    fs = cloud.get_fs(url)

    glob_url = os.path.join(url, TIMESCALE_OUTDIR_TEMPLATE)
    nudged_output_dirs = fs.glob(glob_url)

    nudged_url = _get_path_for_nudging_timescale(
        nudged_output_dirs, nudging_timescale_hr
    )

    datasets = []
    for source in merge_files:
        mapper = fs.get_mapper(os.path.join(nudged_url, f"{source}.zarr"))
        ds = xr.open_zarr(zstore.LRUStoreCache(mapper, 1024))
        times = np.vectorize(round_time)(ds[TIME_NAME])
        ds = ds.assign_coords({TIME_NAME: times})

        datasets.append(ds)

    nudged_mapper = MergeNudged(*datasets)
    nudged_mapper = SubsetTimes(initial_time_skip_hr, n_times, nudged_mapper)

    return nudged_mapper
