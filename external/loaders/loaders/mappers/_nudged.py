import os
import logging
import xarray as xr
import zarr.storage as zstore
from typing import Sequence, Mapping, Union, Tuple, Any
from itertools import product
from toolz import groupby
from pathlib import Path

from vcm import cloud
from ._base import GeoMapper, LongRunMapper
from .._utils import standardize_zarr_time_coord


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


class MergeNudged(LongRunMapper):
    """
    Mapper for merging data sources available from a nudged run.
    
    Currently used to merge the nudging tendencies with the after
    physics checkpointed state information. Could be useful for
    merging prognostic run output by time in the future.
    """

    def __init__(
        self,
        *nudged_sources: Sequence[Union[LongRunMapper, xr.Dataset]],
        rename_vars: Mapping[str, str] = None,
    ):
        rename_vars = rename_vars or {}
        if len(nudged_sources) < 2:
            raise TypeError(
                "MergeNudged should be instantiated with two or more data sources."
            )
        nudged_sources = self._mapper_to_datasets(nudged_sources)
        self._check_dvar_overlap(*nudged_sources)
        self.ds = xr.merge(nudged_sources, join="inner").rename(rename_vars)

    @staticmethod
    def _mapper_to_datasets(data_sources) -> Mapping[str, xr.Dataset]:

        datasets = []
        for source in data_sources:
            if isinstance(source, LongRunMapper):
                source = source.ds
            datasets.append(standardize_zarr_time_coord(source))

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
    Uses LongRunMappers for individual sources.
    """

    def __init__(self, ds_map: Mapping[str, xr.Dataset]):

        self.sources = {key: LongRunMapper(ds) for key, ds in ds_map.items()}

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


class NudgedFullTendencies(GeoMapper):
    def __init__(
        self,
        nudged_mapper: Mapping[Time, xr.Dataset],
        checkpoint_mapper: Mapping[Checkpoint, xr.Dataset],
        difference_checkpoints: Sequence[str] = ("after_dynamics", "after_physics"),
        tendency_variables: Mapping[str, str] = None,
        physics_timestep_seconds: int = 900,
    ):
        for source in difference_checkpoints:
            if source not in checkpoint_mapper.sources:
                raise KeyError(
                    f"Sources must include {' '.join(difference_checkpoints)}"
                    f"but {source} is not present."
                )
        self._nudged_mapper = nudged_mapper
        self._checkpoint_mapper = checkpoint_mapper
        self._difference_checkpoints = difference_checkpoints
        self._tendency_variables = tendency_variables or {
            "pQ1": "air_temperature",
            "pQ2": "specific_humidity",
        }
        self._physics_timestep_seconds = physics_timestep_seconds

    def keys(self):
        return self._nudged_mapper.keys()

    def __getitem__(self, time: Time) -> xr.Dataset:
        return self._derived_ds(time)

    def _derived_ds(self, time: Time):

        physics_tendencies = self._physics_tendencies(
            time,
            self._tendency_variables,
            self._checkpoint_mapper,
            self._difference_checkpoints,
            self._physics_timestep_seconds,
        )

        return self._nudged_mapper[time].assign(physics_tendencies)

    @staticmethod
    def _physics_tendencies(
        time: Time,
        tendency_variables: Mapping[str, str],
        checkpoint_mapper: Checkpoint,
        difference_checkpoints: Sequence[str],
        physics_timestep_seconds: Union[int, float],
    ) -> Mapping[str, xr.DataArray]:

        physics_tendencies = {}
        for term_name, variable_name in tendency_variables.items():
            physics_tendencies[term_name] = (
                checkpoint_mapper[(difference_checkpoints[1], time)][variable_name]
                - checkpoint_mapper[(difference_checkpoints[0], time)][variable_name]
            ) / physics_timestep_seconds

        return physics_tendencies


def open_merged_nudged(
    url: str,
    nudging_timescale_hr: Union[int, float],
    merge_files: Tuple[str] = ("after_physics.zarr", "nudging_tendencies.zarr"),
    initial_time_skip_hr: int = 0,
    n_times: int = None,
    rename_vars: Mapping[str, str] = None,
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
        rename_vars (optional): mapping of variables to be renamed; defaults to
            renaming long nudging names to dQ names
    """

    rename_vars = rename_vars or {
        "air_temperature_tendency_due_to_nudging": "dQ1",
        "specific_humidity_tendency_due_to_nudging": "dQ2",
    }

    fs = cloud.get_fs(url)

    glob_url = os.path.join(url, TIMESCALE_OUTDIR_TEMPLATE)
    nudged_output_dirs = fs.glob(glob_url)

    nudged_url = _get_path_for_nudging_timescale(
        nudged_output_dirs, nudging_timescale_hr
    )

    datasets = []
    for source in merge_files:
        mapper = fs.get_mapper(os.path.join(nudged_url, f"{source}"))
        ds = xr.open_zarr(zstore.LRUStoreCache(mapper, 1024))

        datasets.append(ds)

    nudged_mapper = MergeNudged(*datasets, rename_vars=rename_vars)
    nudged_mapper = SubsetTimes(initial_time_skip_hr, n_times, nudged_mapper)

    return nudged_mapper


def _open_nudging_checkpoints(
    url: str,
    nudging_timescale_hr: Union[int, float],
    checkpoint_files: Tuple[str] = (
        "before_dynamics.zarr",
        "after_dynamics.zarr",
        "after_physics.zarr",
        "after_nudging.zarr",
    ),
) -> Mapping[Checkpoint, xr.Dataset]:
    """
    Load mapper to all checkpoint states and timesteps of a nudging simulation.

    Args:
        url: Path to directory with nudging output (not including the timescale
            subdirectories, e.g., outdir-3h)
        timescale_hours: timescale of the nudging for the simulation
            being used as input.
        checkpoint_files (optionsl): nudged simulation checkpoint files to load
            into the NudgedStateCheckpoints object
    """

    fs = cloud.get_fs(url)

    glob_url = os.path.join(url, TIMESCALE_OUTDIR_TEMPLATE)
    nudged_output_dirs = fs.glob(glob_url)

    nudged_url = _get_path_for_nudging_timescale(
        nudged_output_dirs, nudging_timescale_hr
    )

    datasets = {}
    for filename in checkpoint_files:
        full_path = os.path.join(nudged_url, f"{filename}")
        mapper = fs.get_mapper(full_path)
        ds = xr.open_zarr(zstore.LRUStoreCache(mapper, 1024))

        source_name = Path(filename).stem
        datasets[source_name] = ds

    return NudgedStateCheckpoints(datasets)


def open_merged_nudged_full_tendencies(
    url: str,
    nudging_timescale_hr: Union[int, float],
    open_merged_nudged_kwargs: Mapping[str, Any] = None,
    open_checkpoints_kwargs: Mapping[str, Any] = None,
    difference_checkpoints: Sequence[str] = ("after_dynamics", "after_physics"),
    tendency_variables: Mapping[str, str] = None,
    timestep_physics_seconds: int = 900,
) -> Mapping[str, xr.Dataset]:
    """
    Load mapper to nudged dataset containing both dQ and pQ tendency terms

    Args:
        url: Path to directory with nudging output (not including the timescale
            subdirectories, e.g., outdir-3h)
        timescale_hours: timescale of the nudging for the simulation
            being used as input.
        open_merged_nudged_kwargs (optional): kwargs mapping to be passed to
            open_merged_nudged
        open_checkpoints_kwargs (optional): kwargs mapping to be passed to
            open_nudging_checkpoints
        difference_checkpoints (optional): len-2 sequence of checkpoint names
            for computing physics tendencies, with first checkpoint subtracted
            from second; defaults to ('after_dynamics', 'after_physics')
        tendency_variables (optional): mapping of tendency term names to underlying
            variable names; defaults to
            {'pQ1': 'air_temperature', 'pQ2': 'specific_humidity'}
        timestep_physics_seconds (optional): physics timestep in seconds;
            defaults to 900
        
    Returns
        mapper of timestamps to datasets containing full tendency terms
    """

    open_merged_nudged_kwargs = open_merged_nudged_kwargs or {}
    open_checkpoints_kwargs = open_checkpoints_kwargs or {}

    nudged_mapper = open_merged_nudged(
        url, nudging_timescale_hr, **open_merged_nudged_kwargs
    )

    checkpoint_mapper = _open_nudging_checkpoints(
        url, nudging_timescale_hr, **open_checkpoints_kwargs
    )

    return NudgedFullTendencies(
        nudged_mapper,
        checkpoint_mapper,
        difference_checkpoints,
        tendency_variables,
        timestep_physics_seconds,
    )
