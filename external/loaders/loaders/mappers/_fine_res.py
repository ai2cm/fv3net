import dataclasses
from datetime import timedelta
from enum import Enum
from typing_extensions import Protocol
from typing import Optional, Sequence, Tuple
import zarr
import xarray as xr
import numpy as np
import fsspec

from vcm.fv3.metadata import gfdl_to_standard
from loaders._config import mapper_functions
from loaders.mappers._base import GeoMapper
from loaders.mappers._xarray import XarrayMapper
from loaders.mappers._fine_res_budget import compute_fine_res_sources, FineResBudget
import vcm


class MLTendencies(Protocol):
    dQ1: xr.DataArray
    dQ2: xr.DataArray


class NudgedRun(Protocol):
    tendency_of_air_temperature_due_to_dynamics: xr.DataArray
    tendency_of_specific_humidity_due_to_dynamics: xr.DataArray
    air_temperature_tendency_due_to_nudging: xr.DataArray
    specific_humidity_tendency_due_to_nudging: xr.DataArray


class DynamicsDifferences(Protocol):
    fine_minus_coarse_difference_of_t_dt_dynamics: xr.DataArray
    fine_minus_coarse_difference_of_qv_dt_dynamics: xr.DataArray


class MergedData(NudgedRun, FineResBudget):
    pass


class MergedDataWithDynamicsDifferences(MergedData, DynamicsDifferences):
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

    # enforce that these ML inputs come from fine dataset if they exist there
    if "T" in fine:
        merged["air_temperature"] = fine.T
    if "sphum" in fine:
        merged["specific_humidity"] = fine.sphum

    return merged


class Approach(Enum):
    apparent_sources_only = 1
    apparent_sources_plus_nudging_tendencies = 2
    apparent_sources_extend_lower = 4
    dynamics_difference = 5
    dynamics_difference_flux = 6


@dataclasses.dataclass
class DynamicsDifferenceFluxApparentSource:
    """
    Q  = flux_fit(high_res dyn - coarse dyn) + high_res physics
       = high res (storage - nudge - physics) + high_res physics - coarse dyn
       = high-res storage - high res nudging - coarse dyn tendency
    """

    include_temperature_nudging: bool

    def temperature_source(self, merged: MergedDataWithDynamicsDifferences):
        return (
            merged.Q1 + merged.fine_minus_coarse_difference_of_t_dt_dynamics
        ).assign_attrs(units="K/s")

    def moisture_source(self, merged: MergedDataWithDynamicsDifferences):
        return (
            merged.Q2 + merged.fine_minus_coarse_difference_of_qv_dt_dynamics
        ).assign_attrs(units="kg/kg/s")


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


def _dynamics_tendency_differences(
    merged: MergedData,
) -> Tuple[xr.DataArray, xr.DataArray]:
    T_high_res_dyn = (
        merged.T_storage - merged.t_dt_nudge_coarse - merged.t_dt_phys_coarse
    )
    dyn_diff_T = T_high_res_dyn - merged.tendency_of_air_temperature_due_to_dynamics
    qv_high_res_dyn = merged.sphum_storage - merged.qv_dt_phys_coarse
    dyn_diff_qv = qv_high_res_dyn - merged.tendency_of_specific_humidity_due_to_dynamics
    return dyn_diff_T, dyn_diff_qv


def _dynamics_tendency_differences_as_flux(merged: MergedDataWithDynamicsDifferences):
    qv_g_flux = vcm.fit_field_as_flux(
        merged.fine_minus_coarse_difference_of_qv_dt_dynamics,
        pressure_thickness=merged.delp,
        first_level_flux=first_level_flux,
        last_level_flux=0.0,
        vertical_dim="pfull",
    )


def compute_budget(
    merged: xr.Dataset, approach: Approach, include_temperature_nudging: bool
) -> MLTendencies:
    sources = compute_fine_res_sources(merged, include_temperature_nudging)
    merged = xr.merge([merged] + list(sources))

    (
        merged["fine_minus_coarse_difference_of_t_dt_dynamics"],
        merged["fine_minus_coarse_difference_of_qv_dt_dynamics"],
    ) = _dynamics_tendency_differences(merged)

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

    return _ml_standard_names(merged)


def _add_nudging_tendencies(merged: MergedData):
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


def _ml_standard_names(merged: xr.Dataset):

    # since ML target is Q1/Q2, dQ1=Q1 and same for moistening
    merged["dQ1"] = merged["Q1"]
    merged["dQ2"] = merged["Q2"]

    return merged.astype(np.float32)


@mapper_functions.register
def open_fine_resolution(
    approach: str,
    fine_url: str,
    include_temperature_nudging: bool = False,
    additional_dataset_urls: Sequence[str] = None,
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
        
    Returns:
        a mapper
    """

    approach_enum = Approach[approach]

    merged: FineResBudget = _open_merged_dataset(
        fine_url=fine_url, additional_dataset_urls=additional_dataset_urls,
    )
    budget: MLTendencies = compute_budget(
        merged, approach_enum, include_temperature_nudging=include_temperature_nudging
    )

    return XarrayMapper(budget)


def _open_precomputed_fine_resolution_dataset(
    fine_url: str, additional_dataset_urls: Optional[Sequence[str]] = None
) -> MLTendencies:

    merged = _open_merged_dataset(
        fine_url=fine_url,
        additional_dataset_urls=additional_dataset_urls,
        standardize_fine_coords=False,
    )

    return _ml_standard_names(merged)


@mapper_functions.register
def open_precomputed_fine_resolution(
    fine_url: str, additional_dataset_urls: str = None
) -> GeoMapper:
    """
    Open a fine-res mapper from precomputed data, optionally using state
        from another run.
        
    Args:
        fine_url: url where coarsened fine resolution data is stored, must include
            precomputed Q1 and Q2
        additional_dataset_urls: sequence of urls which to zarrs containing additional
            data to be merged into the resulting mapper dataset
    Returns:
        a mapper
    """
    return XarrayMapper(
        _open_precomputed_fine_resolution_dataset(
            fine_url=fine_url, additional_dataset_urls=additional_dataset_urls
        )
    )
