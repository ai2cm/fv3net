from datetime import timedelta
from enum import Enum
from typing_extensions import Protocol
from typing import Optional, Sequence
import xarray as xr
import numpy as np
import fsspec

from vcm.fv3.metadata import gfdl_to_standard
from loaders._config import mapper_functions
from loaders.mappers._base import GeoMapper
from loaders.mappers._xarray import XarrayMapper
from loaders.mappers._fine_res_budget import compute_fine_res_sources, FineResBudget

class MLTendencies(Protocol):
    dQ1: xr.DataArray
    dQ2: xr.DataArray


def open_zarr(url, consolidated=False):
    mapper = fsspec.get_mapper(url)
    return xr.open_zarr(mapper, consolidated=consolidated)


def open_zarr_maybe_consolidated(url):
    try:
        return open_zarr(url, consolidated=True)
    except KeyError:
        return open_zarr(url, consolidated=False)


def standardize_coords(
    ds: xr.Dataset, time_shift=-timedelta(minutes=7, seconds=30)
) -> xr.Dataset:
    ds_shifted = ds.assign(time=ds.time + time_shift)
    return gfdl_to_standard(ds_shifted).drop("tile")


def _open_fine_resolution_dataset(
    fine_url: str, additional_dataset_urls: Optional[Sequence[str]]
) -> FineResBudget:

    fine = open_zarr_maybe_consolidated(fine_url)
    fine_shifted = standardize_coords(fine)

    if additional_dataset_urls is not None:
        additional_datasets = []
        for url in additional_dataset_urls:
            additional_datasets.append(open_zarr_maybe_consolidated(url))
        merged = xr.merge([fine_shifted, *additional_datasets], join="inner")
        if "latitude" in merged:
            merged["latitude"] = merged.latitude.isel(time=0)
        if "longitude" in merged:
            merged["longitude"] = merged.longitude.isel(time=0)
    else:
        merged = fine_shifted

    return merged


class Approach(Enum):
    apparent_sources_only = 1
    apparent_sources_plus_nudging_tendencies = 2
    apparent_sources_plus_dynamics_differences = 3
    apparent_sources_extend_lower = 4


def _compute_budget(
    merged: xr.Dataset, approach: Approach, include_temperature_nudging: bool
) -> MLTendencies:

    merged["Q1"], merged["Q2"] = compute_fine_res_sources(
        merged, include_temperature_nudging
    )

    if approach == Approach.apparent_sources_plus_nudging_tendencies:
        merged["Q1"], merged["Q2"] = _add_nudging_tendencies(merged)
    elif approach == Approach.apparent_sources_plus_dynamics_differences:
        merged["Q1"], merged["Q2"] = _add_dynamics_differences(merged)
    elif approach == Approach.apparent_sources_extend_lower:
        merged["Q1"] = _extend_lower(merged["Q1"])
        merged["Q2"] = _extend_lower(merged["Q2"])

    return _ml_standard_names(merged)


def _add_nudging_tendencies(merged: xr.Dataset):
    with xr.set_options(keep_attrs=True):
        Q1 = merged.Q1 + merged.air_temperature_tendency_due_to_nudging
        Q2 = merged.Q2 + merged.specific_humidity_tendency_due_to_nudging
    Q1.attrs.update(
        {
            "long_name": merged.Q1.attrs.get("long_name")
            + " plus dynamics nudging tendencies",
            "description": merged.Q1.attrs.get("description")
            + " + dynamics nudging tendencies",
        }
    )
    Q2.attrs.update(
        {
            "long_name": merged.Q2.attrs.get("long_name")
            + " plus dynamics nudging tendencies",
            "description": merged.Q2.attrs.get("description")
            + " + dynamics nudging tendencies",
        }
    )
    return Q1, Q2


def _add_dynamics_differences(merged: xr.Dataset):
    with xr.set_options(keep_attrs=True):
        Q1 = merged.Q1 + merged.fine_minus_coarse_tendency_of_air_temperature_due_to_dynamics
        Q2 = merged.Q2 + merged.fine_minus_coarse_tendency_of_specific_humidity_due_to_dynamics
    Q1.attrs.update(
        {
            "long_name": merged.Q1.attrs.get("long_name")
            + " plus dynamics differences",
            "description": merged.Q1.attrs.get("description")
            + " + dynamics differences",
        }
    )
    Q2.attrs.update(
        {
            "long_name": merged.Q2.attrs.get("long_name")
            + " plus dynamics differences",
            "description": merged.Q2.attrs.get("description")
            + " + dynamics differences",
        }
    )
    return Q1, Q2


def _extend_lower(
    fine_source: xr.DataArray, vertical_dim: str = "z"
) -> xr.Dataset:
    if fine_source.sizes[vertical_dim] < 2:
        raise ValueError("vertical_dim must be greater than 1.")
    fine_source_new_bottom = fine_source.isel({vertical_dim: -2})
    fine_source_without_bottom = fine_source.isel({vertical_dim: slice(None, -1)})
    fine_source_extended_lower = xr.concat(
        [fine_source_without_bottom, fine_source_new_bottom], dim=vertical_dim
    )
    fine_source_extended_lower.attrs.update(
        {
            "long_name": fine_source.attrs.get("long_name")
            + " with lowest layer overriden",
            "description": fine_source.attrs.get("description")
            + ", with lowest layer overriden",
        }
    )
    return fine_source_extended_lower


def _ml_standard_names(merged: xr.Dataset):

    # since ML target is Q1/Q2, dQ1=Q1 and pQ1=0 and same for moistening
    merged["dQ1"] = merged["Q1"]
    merged["dQ2"] = merged["Q2"]
    merged["pQ1"] = xr.zeros_like(merged.Q1)
    merged["pQ1"].attrs = {"units": "K/s", "long_name": "coarse-res physics heating"}
    merged["pQ2"] = xr.zeros_like(merged.Q2)
    merged["pQ2"].attrs = {
        "units": "kg/kg/s",
        "long_name": "coarse-res physics moistening",
    }

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

    merged: FineResBudget = _open_fine_resolution_dataset(
        fine_url=fine_url, additional_dataset_urls=additional_dataset_urls,
    )
    budget: MLTendencies = _compute_budget(
        merged, approach_enum, include_temperature_nudging=include_temperature_nudging
    )

    return XarrayMapper(budget)


def _open_precomputed_fine_resolution_dataset(
    fine_url: str, additional_dataset_urls: Optional[Sequence[str]] = None
) -> MLTendencies:

    merged = _open_fine_resolution_dataset(fine_url, additional_dataset_urls)
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
