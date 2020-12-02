import logging
import xarray as xr
import intake
import os
from typing import Hashable, Sequence, Mapping, Optional, Tuple

from ._common import (
    _get_source_datasets,
    MergeNudged,
    SubtractNudgingTendency,
)
from .._transformations import SubsetTimes
from .._base import MultiDatasetMapper
from .._xarray import XarrayMapper

logger = logging.getLogger(__name__)

Z_DIM_NAME = "z"

Time = str


def open_nudge_to_obs(
    url: str,
    merge_files: Tuple[str] = (
        "data.zarr",
        "physics_tendencies.zarr",
        "nudging_tendencies.zarr",
    ),
    nudging_to_physics_tendency: Mapping[str, str] = None,
    rename_vars: Mapping[str, str] = None,
    consolidated: bool = True,
    i_start: int = 0,
    n_times: int = None,
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
        merge_files: zarrs to merge. Expecting one to contain feature variables,
            one to contain nudging tendencies, and one to contain the tendencies
            across the physics step. Defaults to ("data.zarr",
            "physics_tendencies.zarr", "nudging_tendencies.zarr").
        nudging_to_physics_tendency: Mapping of renamed nudging tendency
            names to physics tendency names; defaults to {'dQ1': 'pQ1, 'dQ2': 'pQ2'}
        rename_vars: Mapping of variables to be renamed. Defaults to {
            "tendency_of_air_temperature_due_to_fv3_physics": "pQ1",
            "tendency_of_specific_humidity_due_to_fv3_physics": "pQ2",
            "t_dt_nudge": "dQ1",
            "q_dt_nudge": "dQ2",
            "u_dt_nudge": "dQu",
            "v_dt_nudge": "dQv",
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
        "u_dt_nudge": "dQu",
        "v_dt_nudge": "dQv",
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
    nudged_mapper = SubtractNudgingTendency(nudged_mapper, nudging_to_physics_tendency)
    return SubsetTimes(i_start, n_times, nudged_mapper)


def open_nudge_to_fine_multiple_datasets(
    urls: Sequence[str], names: Optional[Sequence[Hashable]] = None, **kwargs
):
    """
    Load sequence of mappers to nudged datasets containing dQ tendency terms.

    Args:
        urls: paths to directories with nudging output
        names: sequence of names to assign to the dataset coordinate (optional)
        **kwargs: keyword arguments passed to open_nudge_to_fine

    Returns
        mapper of timestamps to dataset containing tendency terms with a dataset
        dimension
    """
    mappers = [open_nudge_to_fine(url, **kwargs) for url in urls]
    return MultiDatasetMapper(mappers, names=names)


def open_nudge_to_fine(url: str, consolidated=True) -> xr.Dataset:
    """
    Opens a nudge-to-fine rundir as a merged xarray dataset, rather than as a mapper
    
    Args:
        url (str):  path to nudge-to-fine rundir, remote or local
        consolidated (bool): whether zarrs to open have consolidated metadata
        
    Returns:
        xarray dataset of combined nudging tendencies, physics tendencies,
            and model state data
    """

    rename_vars = {
        "air_temperature_tendency_due_to_nudging": "dQ1",
        "specific_humidity_tendency_due_to_nudging": "dQ2",
        "x_wind_tendency_due_to_nudging": "dQxwind",
        "y_wind_tendency_due_to_nudging": "dQywind",
        "tendency_of_air_temperature_due_to_fv3_physics": "pQ1",
        "tendency_of_specific_humidity_due_to_fv3_physics": "pQ2",
    }

    physics_tendencies_zarr = os.path.join(url, "physics_tendencies.zarr")
    nudging_tendencies_zarr = os.path.join(url, "nudge_to_fine_tendencies.zarr")
    state_before_nudging_zarr = os.path.join(url, "state_before_nudging.zarr")

    physics_tendencies = intake.open_zarr(
        physics_tendencies_zarr, consolidated=consolidated
    ).to_dask()
    nudging_tendencies = intake.open_zarr(
        nudging_tendencies_zarr, consolidated=consolidated
    ).to_dask()
    state_before_nudging = intake.open_zarr(
        state_before_nudging_zarr, consolidated=consolidated
    ).to_dask()

    return XarrayMapper(
        xr.merge([state_before_nudging, physics_tendencies, nudging_tendencies]).rename(
            rename_vars
        )
    )
