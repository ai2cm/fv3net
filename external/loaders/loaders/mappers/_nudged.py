import os
import logging
from functools import partial
import xarray as xr
import fsspec
import zarr.storage as zstore
from typing import Sequence, Mapping, Union, Tuple, Any
from itertools import product
from toolz import groupby
from pathlib import Path

import vcm

from ._transformations import KeyMap
from ._base import GeoMapper, LongRunMapper
from ._merged import MergeOverlappingData
from ._high_res_diags import open_high_res_diags
from .._utils import standardize_zarr_time_coord, assign_net_physics_terms
from ..constants import DERIVATION_SHIELD_COORD, DERIVATION_FV3GFS_COORD

logger = logging.getLogger(__name__)

Z_DIM_NAME = "z"


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
        nudged_sources = self._rename_vars(nudged_sources, rename_vars)
        self._check_dvar_overlap(*nudged_sources)
        self.ds = xr.merge(nudged_sources, join="inner")

    @staticmethod
    def _rename_vars(
        datasets: Sequence[xr.Dataset], rename_vars: Mapping[str, str]
    ) -> Sequence[xr.Dataset]:
        renamed_datasets = []
        for ds in datasets:
            ds_rename_vars = {k: v for k, v in rename_vars.items() if k in ds}
            renamed_datasets.append(ds.rename(ds_rename_vars))
        return renamed_datasets

    @staticmethod
    def _mapper_to_datasets(
        data_sources: Sequence[Union[LongRunMapper, xr.Dataset]]
    ) -> Sequence[xr.Dataset]:

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
        return set(keys)


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
        i_start: int,
        n_times: Union[int, None],
        nudged_data: Mapping[str, xr.Dataset],
    ):
        timestep_keys = list(nudged_data.keys())
        timestep_keys.sort()

        i_end = None if n_times is None else i_start + n_times
        self._keys = timestep_keys[slice(i_start, i_end)]
        self._nudged_data = nudged_data

    def keys(self):
        return set(self._keys)

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

        return (
            self._nudged_mapper[time]
            .assign(physics_tendencies)
            .pipe(assign_net_physics_terms)
        )

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


class SubtractNudgingIncrement(GeoMapper):
    """Subtract nudging increment (i.e. nudging tendency times physics timestep) from
    given state."""

    def __init__(
        self,
        nudged_mapper: Mapping[Time, xr.Dataset],
        nudging_tendency_variables: Mapping[str, str],
        physics_timestep_seconds: int,
    ):
        self._nudged_mapper = nudged_mapper
        self._nudging_tendency_variables = nudging_tendency_variables
        self._physics_timestep_seconds = physics_timestep_seconds

    def keys(self):
        return self._nudged_mapper.keys()

    def __getitem__(self, time: Time) -> xr.Dataset:
        return self._derived_ds(time)

    def _derived_ds(self, time: Time):
        before_nudging_state = self._before_nudging_state(time,)
        return self._nudged_mapper[time].assign(before_nudging_state)

    def _before_nudging_state(self, time: Time) -> Mapping[str, xr.DataArray]:
        before_nudging_state = {}
        for variable_name, nudging_name in self._nudging_tendency_variables.items():
            before_nudging_state[variable_name] = (
                self._nudged_mapper[time][variable_name]
                - self._physics_timestep_seconds
                * self._nudged_mapper[time][nudging_name]
            )
        return before_nudging_state


class SubtractNudgingTendency(GeoMapper):
    """Subtract nudging tendency from physics tendency. Necessary for nudge-to-obs."""

    def __init__(
        self,
        nudged_mapper: Mapping[Time, xr.Dataset],
        nudging_to_physics_tendency: Mapping[str, str],
    ):
        self._nudged_mapper = nudged_mapper
        self._nudging_to_physics_tendency = nudging_to_physics_tendency

    def keys(self):
        return self._nudged_mapper.keys()

    def __getitem__(self, time: Time) -> xr.Dataset:
        return self._derived_ds(time)

    def _derived_ds(self, time: Time):
        differenced_physics_tendency = self._subtract_nudging_tendency(time)
        return self._nudged_mapper[time].assign(differenced_physics_tendency)

    def _subtract_nudging_tendency(self, time: Time) -> Mapping[str, xr.DataArray]:
        differenced_physics_tendency = {}
        for nudging_name, physics_name in self._nudging_to_physics_tendency.items():
            differenced_physics_tendency[physics_name] = (
                self._nudged_mapper[time][physics_name]
                - self._nudged_mapper[time][nudging_name]
            )
        return differenced_physics_tendency


def open_merged_nudged(
    url: str,
    merge_files: Tuple[str] = ("after_physics.zarr", "nudging_tendencies.zarr"),
    i_start: int = 0,
    n_times: int = None,
    rename_vars: Mapping[str, str] = None,
    consolidated: bool = False,
) -> Mapping[str, xr.Dataset]:
    """
    Load nudging data mapper for use with training.  Currently merges the
    two files after_physics and nudging_tendencies, which are required I/O
    for the training

    Args:
        url: Path to directory with nudging output (not including the timescale
            subdirectories, e.g., outdir-3h)
        merge_files (optionsl): underlying nudging zarr datasets to combine
            into a MergeNudged mapper
        i_start (optional): Index of sorted timesteps at which to start including
            data in the batch resampling operation; defaults to 0
        n_times (optional): Number of sorted times (by index) to include in the
            batch resampling operation, starting with i_start and ending at
            (i_start + n_times)
        rename_vars (optional): mapping of variables to be renamed; defaults to
            renaming long nudging names to dQ names
        consolidated: if true, open the underlying zarr stores with the consolidated
            flag to xr.open_zarr.
    """

    rename_vars = rename_vars or {
        "air_temperature_tendency_due_to_nudging": "dQ1",
        "specific_humidity_tendency_due_to_nudging": "dQ2",
        "x_wind_tendency_due_to_nudging": "dQx",
        "y_wind_tendency_due_to_nudging": "dQy"
    }

    datasets = _get_source_datasets(url, merge_files, consolidated)
    nudged_mapper = MergeNudged(*datasets, rename_vars=rename_vars)
    nudged_mapper = SubsetTimes(i_start, n_times, nudged_mapper)

    return nudged_mapper


def _open_nudging_checkpoints(
    url: str,
    checkpoint_files: Tuple[str] = (
        "before_dynamics.zarr",
        "after_dynamics.zarr",
        "after_physics.zarr",
        "after_nudging.zarr",
    ),
    consolidated: bool = False,
) -> Mapping[Checkpoint, xr.Dataset]:
    """
    Load mapper to all checkpoint states and timesteps of a nudging simulation.

    Args:
        url: Path to directory with nudging output
        checkpoint_files: nudged simulation checkpoint files to load
            into the NudgedStateCheckpoints object
        consolidated: if true, open the underlying zarr stores with the consolidated
            flag to xr.open_zarr.
    """

    datasets = {}
    for filename in checkpoint_files:
        full_path = os.path.join(url, f"{filename}")
        mapper = fsspec.get_mapper(full_path)
        ds = xr.open_zarr(
            zstore.LRUStoreCache(mapper, 1024),
            consolidated=consolidated,
            mask_and_scale=False,
        )

        source_name = Path(filename).stem
        datasets[source_name] = ds

    return NudgedStateCheckpoints(datasets)


def open_merged_nudged_full_tendencies(
    url: str,
    shield_diags_url: str = None,
    open_merged_nudged_kwargs: Mapping[str, Any] = None,
    open_checkpoints_kwargs: Mapping[str, Any] = None,
    difference_checkpoints: Sequence[str] = ("after_dynamics", "after_physics"),
    tendency_variables: Mapping[str, str] = None,
    timestep_physics_seconds: int = 900,
    consolidated: bool = False,
    offset_seconds: Union[int, float] = 0,
) -> Mapping[str, xr.Dataset]:
    """
    Load mapper to nudged dataset containing both dQ and pQ tendency terms

    Args:
        url: Path to directory with nudging output (not including the timescale
            subdirectories, e.g., outdir-3h)
        shield_diags_url: path to directory containing a zarr store of SHiELD
            diagnostics coarsened to the nudged model resolution (optional)
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
        consolidated: if true, open the underlying zarr stores with the consolidated
            flag to xr.open_zarr.
        offset_seconds: amount to shift the keys forward by in seconds. For
            example, if the underlying data contains a value at the key
            "20160801.000730", a value off 450 will shift this forward 7:30
            minutes, so that this same value can be accessed with the key
            "20160801.001500"

    Returns
        mapper of timestamps to datasets containing full tendency terms
    """

    open_merged_nudged_kwargs = open_merged_nudged_kwargs or {}
    open_checkpoints_kwargs = open_checkpoints_kwargs or {}

    nudged_mapper = open_merged_nudged(
        url, consolidated=consolidated, **open_merged_nudged_kwargs
    )
    checkpoint_mapper = _open_nudging_checkpoints(
        url, consolidated=consolidated, **open_checkpoints_kwargs
    )

    full_tendencies_mapper = NudgedFullTendencies(
        nudged_mapper,
        checkpoint_mapper,
        difference_checkpoints,
        tendency_variables,
        timestep_physics_seconds,
    )

    full_tendencies_mapper = KeyMap(
        partial(vcm.shift_timestamp, seconds=offset_seconds), full_tendencies_mapper,
    )

    if shield_diags_url is not None:
        shield_diags_mapper = open_high_res_diags(shield_diags_url)
        full_tendencies_mapper = MergeOverlappingData(
            shield_diags_mapper,
            full_tendencies_mapper,
            source_name_left=DERIVATION_SHIELD_COORD,
            source_name_right=DERIVATION_FV3GFS_COORD,
        )

    return full_tendencies_mapper


def open_merged_nudge_to_obs(
    url: str,
    merge_files: Tuple[str] = ("after_physics.zarr", "nudging_tendencies.zarr"),
    i_start: int = 0,
    n_times: int = None,
    rename_vars: Mapping[str, str] = None,
    nudging_tendency_variables: Mapping[str, str] = None,
    timestep_physics_seconds: int = 900,
    consolidated: bool = False,
) -> Mapping[str, xr.Dataset]:
    """
    Load nudging data mapper for use with training. Currently merges the
    two files after_physics and nudging_tendencies. Since the nudge-to-obs run does
    nudging within the physics routines, the nudging increments are subtracted from
    the after_physics state in order to provide the actual "before nudging" state.

    Args:
        url: Path to directory with nudging output
        merge_files (optional): underlying nudging zarr datasets to combine
            into a MergeNudged mapper
        i_start (optional): Index of sorted timesteps at which to start including
            data in the batch resampling operation; defaults to 0
        n_times (optional): Number of sorted times (by index) to include in the
            batch resampling operation, starting with i_start and ending at
            (i_start + n_times)
        rename_vars (optional): mapping of variables to be renamed; defaults to
            {"t_dt_nudge": "dQ1", "q_dt_nudge": "dQ2"}
        nudging_tendency_variables: (optional): mapping of variables to their renamed
            nudging tendencies. Defaults to
            {"air_temperature": "dQ1", "specific_humidity": "dQ2"}
        timestep_physics_seconds: model physics timestep in seconds. Defaults to 900.
            Required in order to subtract the nudging increment from the state.
        consolidated: if true, open the underlying zarr stores with the consolidated
            flag to xr.open_zarr.
    """

    rename_vars = rename_vars or {
        "t_dt_nudge": "dQ1",
        "q_dt_nudge": "dQ2",
    }

    nudging_tendency_variables = nudging_tendency_variables or {
        "air_temperature": "dQ1",
        "specific_humidity": "dQ2",
    }
    datasets = _get_source_datasets(url, merge_files, consolidated)
    nudged_mapper = MergeNudged(*datasets, rename_vars=rename_vars)
    nudged_mapper = SubtractNudgingIncrement(
        nudged_mapper, nudging_tendency_variables, timestep_physics_seconds
    )
    nudged_mapper = SubsetTimes(i_start, n_times, nudged_mapper)

    return nudged_mapper


def open_merged_nudge_to_obs_full_tendencies(
    url: str,
    open_merged_nudge_to_obs_kwargs: Mapping[str, Any] = {},
    open_checkpoints_kwargs: Mapping[str, Any] = {},
    difference_checkpoints: Sequence[str] = ("after_dynamics", "after_physics"),
    physics_tendency_variables: Mapping[str, str] = None,
    nudging_to_physics_tendency: Mapping[str, str] = None,
    timestep_physics_seconds: int = 900,
    consolidated: bool = False,
) -> Mapping[str, xr.Dataset]:
    """
    Load mapper to nudge-to-obs dataset containing both dQ and pQ tendency terms.
    Since the nudge-to-obs run does nudging within the physics routines, the physics
    tendency is equal to the difference between the after_dynamics and after_physics
    checkpoints minus the nudging tendency.
    Args:
        url: Path to directory with nudging output
        open_merged_nudge_to_obs_kwargs (optional): kwargs mapping to be passed to
            open_merged_nudge_to_obs
        open_checkpoints_kwargs (optional): kwargs mapping to be passed to
            open_nudging_checkpoints
        difference_checkpoints (optional): len-2 sequence of checkpoint names
            for computing physics tendencies, with first checkpoint subtracted
            from second; defaults to ('after_dynamics', 'after_physics')
        physics_tendency_variables (optional): mapping of tendency term names to
            variable names; defaults to
            {'air_temperature': 'pQ1', 'specific_humidity': 'pQ2'}
        nudging_to_physics_tendency (optional): mapping of renamed nudging tendency
            names to physics tendency names; defaults to {'dQ1': 'pQ1, 'dQ2': 'pQ2'}
        timestep_physics_seconds (optional): physics timestep in seconds;
            defaults to 900
        consolidated (optional): if true, open the underlying zarr stores with the
            consolidated flag to xr.open_zarr. Defaults to false.
    Returns
        mapper of timestamps to datasets containing full tendency terms
    """

    physics_tendency_variables = physics_tendency_variables or {
        "pQ1": "air_temperature",
        "pQ2": "specific_humidity",
    }

    nudging_to_physics_tendency = nudging_to_physics_tendency or {
        "dQ1": "pQ1",
        "dQ2": "pQ2",
    }

    nudged_mapper = open_merged_nudge_to_obs(
        url, consolidated=consolidated, **open_merged_nudge_to_obs_kwargs
    )
    checkpoint_mapper = _open_nudging_checkpoints(
        url, consolidated=consolidated, **open_checkpoints_kwargs
    )

    full_tendencies_mapper = NudgedFullTendencies(
        nudged_mapper,
        checkpoint_mapper,
        difference_checkpoints,
        physics_tendency_variables,
        timestep_physics_seconds,
    )

    full_tendencies_mapper = SubtractNudgingTendency(
        full_tendencies_mapper, nudging_to_physics_tendency
    )

    return full_tendencies_mapper


def open_nudged_to_obs_prognostic(
    url: str,
    merge_files: Tuple[str] = ("data.zarr", "nudging_tendencies.zarr"),
    nudging_to_physics_tendency: Mapping[str, str] = None,
    rename_vars: Mapping[str, str] = None,
    consolidated: bool = False,
) -> Mapping[str, xr.Dataset]:
    """Load nudging data mapper for use with training. Merges the
    data variables saved in the diagnostics files (fv3config[diagnostics][variables])
    and nudging_tendencies. Since the nudge-to-obs run does
    nudging within the physics routines, the nudging tendencies are subtracted from
    the tendencies across the physics step to obtain the tendencies from
    model physics.

    Note the difference between this function and open_merged_nudge_to_obs:
    in the prognostic nudge to obs data, the tendency across the physics step
    is already calculated.

    Args:
        url: Path to directory containing merge_files. Defaults to str.
        merge_files: zarrs to merge. Expecting one to contain nudging tendencies
            and the other to contain the tendencies across the physics step.
            Defaults to ("data.zarr", "nudging_tendencies.zarr").
        nudging_to_physics_tendency: Mapping of renamed nudging tendency
            names to physics tendency names; defaults to {'dQ1': 'pQ1, 'dQ2': 'pQ2'}
        rename_vars: Mapping of variables to be renamed. Defaults to {
            "tendency_of_air_temperature_due_to_fv3_physics": "pQ1",
            "tendency_of_specific_humidity_due_to_fv3_physics": "pQ2",
            "t_dt_nudge": "dQ1",
            "q_dt_nudge": "dQ2",
            "grid_xt": "x",
            "grid_yt": "y",
            "pfull": "z"}
        consolidated: if true, open the underlying zarr stores with the consolidated
            flag to xr.open_zarr. Defaults to False.

    Returns:
        Mapper that has the pQ's from only model physics.
    """

    rename_vars = rename_vars or {
        "tendency_of_air_temperature_due_to_fv3_physics": "pQ1",
        "tendency_of_specific_humidity_due_to_fv3_physics": "pQ2",
        "t_dt_nudge": "dQ1",
        "q_dt_nudge": "dQ2",
        "grid_xt": "x",
        "grid_yt": "y",
        "pfull": "z",
    }
    nudging_to_physics_tendency = nudging_to_physics_tendency or {
        "dQ1": "pQ1",
        "dQ2": "pQ2",
    }
    datasets = _get_source_datasets(url, merge_files, consolidated)
    nudged_mapper = MergeNudged(*datasets, rename_vars=rename_vars)
    return SubtractNudgingTendency(nudged_mapper, nudging_to_physics_tendency)


def _get_source_datasets(
    url: str, sources: Sequence[str], consolidated: bool = False
) -> Sequence[xr.Dataset]:
    datasets = []
    for source in sources:
        mapper = fsspec.get_mapper(os.path.join(url, f"{source}"))
        ds = xr.open_zarr(
            zstore.LRUStoreCache(mapper, 1024),
            consolidated=consolidated,
            mask_and_scale=False,
        )
        datasets.append(ds)
    return datasets
