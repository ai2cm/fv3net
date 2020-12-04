import logging
import xarray as xr
import intake
import os
from typing import Hashable, Sequence, Mapping, Optional, Any, MutableMapping

from .._base import MultiDatasetMapper
from .._xarray import XarrayMapper

logger = logging.getLogger(__name__)

Z_DIM_NAME = "z"

Time = str
Dataset = MutableMapping[Hashable, Any]


def open_nudge_to_obs(
    url: str,
    nudging_tendency_variables: Optional[Mapping[str, str]] = None,
    physics_timestep_seconds: float = 900.0,
    consolidated: bool = True,
):
    """
    Load nudge-to-obs data mapper for use with training. Merges
    variables saved in the physics tendencies, nudging tendencies (Fortran
    diagnostics), and model state zarrs.
    
    Because the nudge-to-obs routine conducts nudging within the physics step,
    the returned physics tendency is computed as the output physics_tendency minus
    the nudging tendency. Similarly, because model states are output at the end
    of the timestep, the nudging increment is subtracted to return the
    ``before nudging`` state for training.
    
    Args:
        url (str): path to a nudge-to-obs output directory, remote or local
        nudging_tendency_variables: (optional): mapping of variables to their renamed
            nudging tendencies. Defaults to
            {"air_temperature": "dQ1", "specific_humidity": "dQ2"}
        physics_timestep_seconds (float): physics timestep, i.e., dt_atmos; defaults
            to 900.0
        consolidated (bool): whether zarrs to open have consolidated metadata
        
    Returns:
        mapper to dataset containing nudging tendencies, physics tendencies,
            and model state data
        
    """

    datasets = _get_datasets(
        url,
        [
            "physics_tendencies.zarr",
            "nudge_to_obs_tendencies.zarr",
            "state_after_timestep.zarr",
        ],
        consolidated=consolidated,
    )

    ds = xr.merge(
        [
            datasets["physics_tendencies.zarr"].rename(
                {
                    "tendency_of_air_temperature_due_to_fv3_physics": "pQ1",
                    "tendency_of_specific_humidity_due_to_fv3_physics": "pQ2",
                    "tendency_of_eastward_wind_due_to_fv3_physics": "pQu",
                    "tendency_of_northward_wind_due_to_fv3_physics": "pQv",
                }
            ),
            datasets["nudge_to_obs_tendencies.zarr"].rename(
                {
                    "t_dt_nudge": "dQ1",
                    "q_dt_nudge": "dQ2",
                    "u_dt_nudge": "dQu",
                    "v_dt_nudge": "dQv",
                    "grid_xt": "x",
                    "grid_yt": "y",
                    "pfull": "z",
                }
            ),
            datasets["state_after_timestep.zarr"],
        ]
    )

    nudging_tendency_variables = nudging_tendency_variables or {
        "air_temperature": "dQ1",
        "specific_humidity": "dQ2",
        "eastward_wind": "dQu",
        "northward_wind": "dQv",
    }

    differenced_state: Dataset = {}
    for (
        nudging_variable_name,
        nudging_tendency_name,
    ) in nudging_tendency_variables.items():
        differenced_state[nudging_variable_name] = (
            ds[nudging_variable_name]
            - ds[nudging_tendency_name] * physics_timestep_seconds
        )
    ds = ds.assign(differenced_state)

    differenced_physics_tendency: Dataset = {}
    for nudging_name, physics_name in zip(
        ["dQ1", "dQ2", "dQu", "dQv"], ["pQ1", "pQ2", "pQu", "pQv"]
    ):
        differenced_physics_tendency[physics_name] = ds[physics_name] - ds[nudging_name]
    ds = ds.assign(differenced_physics_tendency)

    return XarrayMapper(ds)


def open_nudge_to_fine(
    url: str,
    nudging_variables: Sequence[str],
    physics_timestep_seconds: float = 900.0,
    consolidated: bool = True,
) -> XarrayMapper:
    """
    Load nudge-to-fine data mapper for use with training. Merges
    variables saved in the physics tendencies, nudging tendencies, and
    model state zarrs.
    
    Because model states are output at the end of the timestep, the nudging
    increment is subtracted to return the ``before nudging`` state for training.
    
    Args:
        url (str):  path to nudge-to-fine output directory, remote or local
        nudging_variables (Sequence[str]): Names of nudged variables
        physics_timestep_seconds (float): physics timestep, i.e., dt_atmos; defaults
            to 900.0
        consolidated (bool): whether zarrs to open have consolidated metadata
        
    Returns:
        mapper to dataset containing nudging tendencies, physics tendencies,
            and model state data
    """

    ds = xr.merge(
        _get_datasets(
            url,
            [
                "physics_tendencies.zarr",
                "nudge_to_fine_tendencies.zarr",
                "state_after_timestep.zarr",
            ],
            consolidated=consolidated,
        ).values()
    )

    differenced_state: Dataset = {}
    for nudging_variable in nudging_variables:
        nudging_tendency = ds[f"{nudging_variable}_tendency_due_to_nudging"]
        differenced_state[nudging_variable] = (
            ds[nudging_variable] - nudging_tendency * physics_timestep_seconds
        )
    ds = ds.assign(differenced_state)

    rename_vars: Mapping[Hashable, Hashable] = {
        "air_temperature_tendency_due_to_nudging": "dQ1",
        "specific_humidity_tendency_due_to_nudging": "dQ2",
        "x_wind_tendency_due_to_nudging": "dQxwind",
        "y_wind_tendency_due_to_nudging": "dQywind",
        "tendency_of_air_temperature_due_to_fv3_physics": "pQ1",
        "tendency_of_specific_humidity_due_to_fv3_physics": "pQ2",
        "tendency_of_eastward_wind_due_to_fv3_physics": "pQu",
        "tendency_of_northward_wind_due_to_fv3_physics": "pQv",
    }

    return XarrayMapper(ds.rename(rename_vars))


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


def _get_datasets(
    url: str, sources: Sequence[str], consolidated: bool = True
) -> MutableMapping[Hashable, xr.Dataset]:
    datasets: MutableMapping[Hashable, xr.Dataset] = {}
    for source in sources:
        ds = intake.open_zarr(
            os.path.join(url, f"{source}"), consolidated=consolidated
        ).to_dask()
        datasets[source] = ds
    return datasets
