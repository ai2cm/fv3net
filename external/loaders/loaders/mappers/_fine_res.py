import dataclasses
from datetime import timedelta
from enum import Enum
from typing_extensions import Protocol
from typing import Optional, Sequence
import zarr
import xarray as xr
import numpy as np
import fsspec

from vcm.fv3.metadata import gfdl_to_standard
from loaders._config import mapper_functions
from loaders.mappers._base import GeoMapper
from loaders.mappers._xarray import XarrayMapper
from loaders.mappers._fine_res_budget import (
    compute_fine_res_sources,
    column_integrated_fine_res_nudging_heating,
    FineResBudget,
    FINE_RES_STATE_NAMES,
    FINE_RES_FLUX_NAMES,
)

COLUMN_T_NUDGE = "storage_of_internal_energy_path_due_to_fine_res_temperature_nudging"
TOA_NET_RADIATION = "total_sky_net_radiative_flux_at_top_of_atmosphere"
TOA_DOWN_SW = "total_sky_downward_shortwave_flux_at_top_of_atmosphere"
TOA_UP_SW = "total_sky_upward_longwave_flux_at_top_of_atmosphere"
TOA_UP_LW = "total_sky_upward_longwave_flux_at_top_of_atmosphere"


class MLTendencies(Protocol):
    dQ1: xr.DataArray
    dQ2: xr.DataArray


class NudgedRun(Protocol):
    tendency_of_air_temperature_due_to_dynamics: xr.DataArray
    tendency_of_specific_humidity_due_to_dynamics: xr.DataArray


class MergedData(NudgedRun, FineResBudget):
    pass


def open_zarr(url, consolidated=False):
    mapper = zarr.LRUStoreCache(fsspec.get_mapper(url), 128 * 2 ** 20)
    return xr.open_zarr(mapper)


def standardize_coords(
    ds: xr.Dataset, time_shift=-timedelta(minutes=7, seconds=30)
) -> xr.Dataset:
    ds_shifted = ds.assign(time=ds.time + time_shift)
    return gfdl_to_standard(ds_shifted).drop("tile")


def _open_merged_dataset(
    fine_url: str,
    additional_dataset_urls: Optional[Sequence[str]],
    standardize_fine_coords: bool = True,
    use_fine_res_state: bool = True,
    use_fine_res_fluxes: bool = False,
) -> FineResBudget:

    fine = open_zarr(fine_url)
    if standardize_fine_coords:
        fine = standardize_coords(fine)

    if additional_dataset_urls is not None:
        additional_datasets = []
        for url in additional_dataset_urls:
            additional_datasets.append(open_zarr(url))
        merged = xr.merge([fine, *additional_datasets], join="inner")
        if "latitude" in merged:
            merged["latitude"] = merged.latitude.isel(time=0)
        if "longitude" in merged:
            merged["longitude"] = merged.longitude.isel(time=0)
    else:
        merged = fine

    # optionally overwrite standard name arrays with those from fine-res budget
    if use_fine_res_state:
        for fine_res_name, standard_name in FINE_RES_STATE_NAMES.items():
            merged[standard_name] = fine[fine_res_name].shift(time=-1)
    if use_fine_res_fluxes:
        for fine_res_name, standard_name in FINE_RES_FLUX_NAMES.items():
            merged[standard_name] = fine[fine_res_name]

    return merged


class Approach(Enum):
    apparent_sources_only = 1
    apparent_sources_plus_nudging_tendencies = 2
    apparent_sources_extend_lower = 4
    dynamics_difference = 5


@dataclasses.dataclass
class DynamicsDifferenceApparentSource:
    """
    Q  = (high_res dyn - coarse dyn) + high_res physics
       = high res (storage - nudge - physics) + high_res physics - coarse dyn
       = high-res storage - high res nudging - coarse dyn tendency
    """

    include_temperature_nudging: bool

    def temperature_source(self, merged: MergedData):
        if self.include_temperature_nudging:
            return (
                merged.T_storage - merged.tendency_of_air_temperature_due_to_dynamics
            ).assign_attrs(units="K/s")
        else:
            return (
                merged.T_storage
                - merged.t_dt_nudge_coarse
                - merged.tendency_of_air_temperature_due_to_dynamics
            ).assign_attrs(units="K/s")

    def moisture_source(self, merged: MergedData):
        return (
            merged.sphum_storage - merged.tendency_of_specific_humidity_due_to_dynamics
        ).assign_attrs(units="kg/kg/s")


def compute_budget(
    merged: xr.Dataset, approach: Approach, include_temperature_nudging: bool
) -> MLTendencies:
    sources = compute_fine_res_sources(merged, include_temperature_nudging)
    merged = xr.merge([merged] + list(sources))

    if approach == Approach.apparent_sources_plus_nudging_tendencies:
        merged["Q1"], merged["Q2"] = _add_nudging_tendencies(merged)
    elif approach == Approach.apparent_sources_extend_lower:
        merged["Q1"] = _extend_lower(merged["Q1"])
        merged["Q2"] = _extend_lower(merged["Q2"])
    elif approach == Approach.dynamics_difference:
        budget = DynamicsDifferenceApparentSource(include_temperature_nudging)
        merged["Q1"] = budget.temperature_source(merged)
        merged["Q2"] = budget.moisture_source(merged)
    elif approach == Approach.apparent_sources_only:
        pass
    else:
        raise ValueError(f"{approach} not implemented.")

    if include_temperature_nudging:
        merged[COLUMN_T_NUDGE] = column_integrated_fine_res_nudging_heating(merged)

    try:
        # older vesions of fine-res budget may not include radiative fluxes
        _compute_net_toa_radiative_flux(merged, include_temperature_nudging)
    except KeyError:
        pass

    return merged.astype(np.float32)


def _add_nudging_tendencies(merged: xr.Dataset):
    with xr.set_options(keep_attrs=True):
        Q1 = merged.Q1 + merged.air_temperature_tendency_due_to_nudging
        Q2 = merged.Q2 + merged.specific_humidity_tendency_due_to_nudging
    Q1.attrs.update(
        {
            "long_name": merged.Q1.attrs.get("long_name")
            + " plus dynamics nudging tendency",
            "description": merged.Q1.attrs.get("description")
            + " + dynamics nudging tendency",
        }
    )
    Q2.attrs.update(
        {
            "long_name": merged.Q2.attrs.get("long_name")
            + " plus dynamics nudging tendency",
            "description": merged.Q2.attrs.get("description")
            + " + dynamics nudging tendency",
        }
    )
    return Q1, Q2


def _extend_lower(
    fine_source: xr.DataArray, vertical_dim: str = "z", n_levels: int = 2
) -> xr.Dataset:
    if fine_source.sizes[vertical_dim] < 2:
        raise ValueError("vertical_dim must be greater than 1.")
    fine_source_new_lower = fine_source.isel({vertical_dim: -(n_levels + 1)})
    fine_source_without_lower = fine_source.isel({vertical_dim: slice(None, -n_levels)})
    fine_source_extended_lower = xr.concat(
        [fine_source_without_lower, *(n_levels * [fine_source_new_lower])],
        dim=vertical_dim,
    )
    fine_source_extended_lower.attrs.update(
        {
            "long_name": fine_source.attrs.get("long_name")
            + " with lowest layer(s) overriden",
            "description": fine_source.attrs.get("description")
            + ", with lowest layer(s) overriden",
        }
    )
    return fine_source_extended_lower


def _compute_net_toa_radiative_flux(
    ds: xr.Dataset, include_temperature_nudging: bool
) -> None:
    ds[TOA_NET_RADIATION] = ds[TOA_DOWN_SW] - ds[TOA_UP_SW] - ds[TOA_UP_LW]
    if include_temperature_nudging:
        ds[TOA_NET_RADIATION] += ds[COLUMN_T_NUDGE]
    ds[TOA_NET_RADIATION].attrs = {
        "long_name": "Net radiative flux at TOA",
        "units": "W/m**2",
    }


@mapper_functions.register
def open_fine_resolution(
    approach: str,
    fine_url: str,
    include_temperature_nudging: bool = False,
    additional_dataset_urls: Sequence[str] = None,
    use_fine_res_state: bool = True,
    use_fine_res_fluxes: bool = False,
) -> GeoMapper:
    """
    Open the fine-res mapper using several configuration options

    Args:
        approach: one of a set of available approaches: 'apparent_sources_only',
            'apparent_sources_plus_nudging_tendencies',
            'apparent_sources_plus_dynamics_differences', or
            'apparent_sources_extend_lower'.
        fine_url: url where coarsened fine resolution data is stored
        include_temperature_nudging: whether to include fine-res nudging in Q1
        additional_dataset_urls: sequence of urls to zarrs containing additional
            data to be merged into the resulting mapper dataset, e.g., ML input
            features, the dynamics nudging tendencies, and the dynamics differences
            as required by the above approaches
        use_fine_res_state: set standard name state variables to point to the fine-res
            data. Set to True if wanting to use fine-res state as ML inputs in training.
        use_fine_res_fluxes: set standard name surface and TOA flux diagnostic variables
            to point to the fine-res data. Set of True if wanting to use fine-res fluxes
            as ML inputs in training.

    Returns:
        a mapper
    """

    approach_enum = Approach[approach]

    merged: FineResBudget = _open_merged_dataset(
        fine_url=fine_url,
        additional_dataset_urls=additional_dataset_urls,
        use_fine_res_state=use_fine_res_state,
        use_fine_res_fluxes=use_fine_res_fluxes,
    )
    budget: MLTendencies = compute_budget(
        merged, approach_enum, include_temperature_nudging=include_temperature_nudging
    )

    return XarrayMapper(budget)
