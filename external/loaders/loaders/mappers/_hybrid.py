import xarray as xr
import numpy as np

from loaders.mappers._base import GeoMapper
from loaders.mappers._xarray import XarrayMapper
from loaders._config import mapper_functions
from loaders.mappers._fine_res import (
    open_zarr,
    standardize_coords,
    MLTendencies,
)
from loaders.mappers._fine_res_budget import compute_fine_res_sources


def _open_fine_resolution_nudging_hybrid_dataset(
    # created by this commit:
    # https://github.com/VulcanClimateModeling/vcm-workflow-control/commit/3c852d0e4f8b86c4e88db9f29f0b8e484aeb77a1
    # I manually consolidated the metadata with zarr.consolidate_metadata
    fine_url: str = "gs://vcm-ml-experiments/default/2021-04-27/2020-05-27-40-day-X-SHiELD-simulation/fine-res-budget.zarr",  # noqa: E501
    # created by this commit
    # https://github.com/VulcanClimateModeling/vcm-workflow-control/commit/dd4498bcf3143d05095bf9ff4ca3f1341ba25330
    nudge_url="gs://vcm-ml-experiments/2021-04-13-n2f-c3072/3-hrly-ave-rad-precip-setting-30-min-rad-timestep-shifted-start-tke-edmf",  # noqa: E501
    include_temperature_nudging: bool = False,
) -> xr.Dataset:

    fine = open_zarr(fine_url)
    fine_shifted = standardize_coords(fine)
    fine_shifted["Q1"], fine_shifted["Q2"] = compute_fine_res_sources(
        fine_shifted, include_temperature_nudging
    )

    return _open_nudged_hybrid_portion(fine_shifted, nudge_url)


def _open_precomputed_fine_resolution_nudging_hybrid_dataset(
    fine_url: str, nudge_url: str,
) -> xr.Dataset:

    fine = open_zarr(fine_url)

    return _open_nudged_hybrid_portion(fine, nudge_url)


def _open_nudged_hybrid_portion(
    fine_shifted: xr.Dataset, nudge_url: str
) -> MLTendencies:

    nudge_physics_tendencies = open_zarr(nudge_url + "/physics_tendencies.zarr",)
    nudge_state = open_zarr(nudge_url + "/state_after_timestep.zarr")
    nudge_tends = open_zarr(nudge_url + "/nudging_tendencies.zarr")

    merged = xr.merge(
        [fine_shifted, nudge_state, nudge_physics_tendencies], join="inner",
    )

    # dQ1,2,u,v
    # "hybrid" definitions for humidity and moisture
    merged["dQ1"] = (
        merged["Q1"] - merged["tendency_of_air_temperature_due_to_fv3_physics"]
    )
    merged["dQ2"] = (
        merged["Q2"] - merged["tendency_of_specific_humidity_due_to_fv3_physics"]
    )
    merged["dQxwind"] = nudge_tends.x_wind_tendency_due_to_nudging
    merged["dQywind"] = nudge_tends.y_wind_tendency_due_to_nudging

    # drop time from lat and lon
    merged["latitude"] = merged.latitude.isel(time=0)
    merged["longitude"] = merged.longitude.isel(time=0)

    return merged.astype(np.float32)


@mapper_functions.register
def open_fine_resolution_nudging_hybrid(
    fine_url: str = "", nudge_url: str = "", include_temperature_nudging: bool = False,
) -> GeoMapper:
    """
    Open the fine resolution nudging_hybrid mapper

    Args:
        fine_url: url where coarsened fine resolution data is stored
        nudge_url: url to nudging data to be used as a residual
        include_temperature_nudging: whether to include fine-res nudging in Q1

    Returns:
        a mapper
    """
    return XarrayMapper(
        _open_fine_resolution_nudging_hybrid_dataset(
            fine_url=fine_url,
            nudge_url=nudge_url,
            include_temperature_nudging=include_temperature_nudging,
        )
    )


@mapper_functions.register
def open_precomputed_fine_resolution_nudging_hybrid(
    fine_url: str, nudge_url: str,
) -> GeoMapper:
    """
    Open the fine resolution nudging hybrid mapper with precomputed fine-res data

    Args:
        fine_url: url where coarsened fine resolution data is stored, must include
            precomputed Q1 and Q2
        nudge_url: url to nudging data to be used as a residual

    Returns:
        a mapper
    """
    return XarrayMapper(
        _open_precomputed_fine_resolution_nudging_hybrid_dataset(
            fine_url=fine_url, nudge_url=nudge_url
        )
    )
