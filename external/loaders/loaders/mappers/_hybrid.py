import fsspec
import xarray
import numpy
import numpy as np
from datetime import timedelta
from typing_extensions import Protocol
from typing import Optional, Tuple, Sequence
from enum import Enum

from loaders.mappers._base import GeoMapper
from loaders.mappers._xarray import XarrayMapper
from loaders._config import mapper_functions
from vcm.fv3.metadata import gfdl_to_standard


def eddy_flux_coarse(unresolved_flux, total_resolved_flux, omega, field):
    """Compute re-coarsened eddy flux divergence from re-coarsed data
    """
    return unresolved_flux + (total_resolved_flux - omega * field)


def _center_to_interface(f: np.ndarray) -> np.ndarray:
    """Interpolate vertically cell centered data to the interface
    with linearly extrapolated inputs"""
    f_low = 2 * f[..., 0] - f[..., 1]
    f_high = 2 * f[..., -1] - f[..., -2]
    pad = np.concatenate([f_low[..., np.newaxis], f, f_high[..., np.newaxis]], axis=-1)
    return (pad[..., :-1] + pad[..., 1:]) / 2


def _convergence(eddy: np.ndarray, delp: np.ndarray) -> np.ndarray:
    """Compute vertical convergence of a cell-centered flux.

    This flux is assumed to vanish at the vertical boundaries
    """
    padded = _center_to_interface(eddy)
    # pad interfaces assuming eddy = 0 at edges
    return -np.diff(padded, axis=-1) / delp


def convergence(
    eddy: xarray.DataArray, delp: xarray.DataArray, dim: str = "p"
) -> xarray.DataArray:
    return xarray.apply_ufunc(
        _convergence,
        eddy,
        delp,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim]],
        dask="parallelized",
        output_dtypes=[eddy.dtype],
    )


class FineResBudget(Protocol):
    """Protocol defining what input vaiables are required

    Only used for type checking and editor autocompletion.
    """

    area: xarray.DataArray
    delp: xarray.DataArray
    T: xarray.DataArray
    dq3dt_deep_conv_coarse: xarray.DataArray
    dq3dt_mp_coarse: xarray.DataArray
    dq3dt_pbl_coarse: xarray.DataArray
    dq3dt_shal_conv_coarse: xarray.DataArray
    dt3dt_deep_conv_coarse: xarray.DataArray
    dt3dt_lw_coarse: xarray.DataArray
    dt3dt_mp_coarse: xarray.DataArray
    dt3dt_ogwd_coarse: xarray.DataArray
    dt3dt_pbl_coarse: xarray.DataArray
    dt3dt_shal_conv_coarse: xarray.DataArray
    dt3dt_sw_coarse: xarray.DataArray
    eddy_flux_vulcan_omega_sphum: xarray.DataArray
    eddy_flux_vulcan_omega_temp: xarray.DataArray
    exposed_area: xarray.DataArray
    qv_dt_fv_sat_adj_coarse: xarray.DataArray
    qv_dt_phys_coarse: xarray.DataArray
    sphum: xarray.DataArray
    sphum_storage: xarray.DataArray
    sphum_vulcan_omega_coarse: xarray.DataArray
    t_dt_fv_sat_adj_coarse: xarray.DataArray
    t_dt_nudge_coarse: xarray.DataArray
    t_dt_phys_coarse: xarray.DataArray
    vulcan_omega_coarse: xarray.DataArray
    T_vulcan_omega_coarse: xarray.DataArray


def open_zarr(url, consolidated=False):
    mapper = fsspec.get_mapper(url)
    return xarray.open_zarr(mapper, consolidated=consolidated)


def open_zarr_maybe_consolidated(url):
    try:
        return open_zarr(url, consolidated=True)
    except KeyError:
        return open_zarr(url, consolidated=False)


def apparent_heating(data: FineResBudget, include_temperature_nudging: bool = False):
    eddy_flux = eddy_flux_coarse(
        data.eddy_flux_vulcan_omega_temp,
        data.T_vulcan_omega_coarse,
        data.vulcan_omega_coarse,
        data.T,
    )
    eddy_flux_convergence = convergence(eddy_flux, data.delp, dim="z")
    result = data.t_dt_fv_sat_adj_coarse + data.t_dt_phys_coarse + eddy_flux_convergence
    description = (
        "Apparent heating due to physics and sub-grid-scale advection. Given "
        "by sat adjustment (dycore) + physics tendency + eddy-flux-convergence"
    )
    if include_temperature_nudging:
        result = result + data.t_dt_nudge_coarse
        description = description + " + temperature nudging"
    return result.assign_attrs(
        units="K/s",
        long_name="apparent heating from high resolution data",
        description=description,
    ).rename("Q1")


def apparent_moistening(data: FineResBudget):
    eddy_flux = eddy_flux_coarse(
        data.eddy_flux_vulcan_omega_sphum,
        data.sphum_vulcan_omega_coarse,
        data.vulcan_omega_coarse,
        data.sphum,
    )
    eddy_flux_convergence = convergence(eddy_flux, data.delp, dim="z")
    return (
        (data.qv_dt_fv_sat_adj_coarse + data.qv_dt_phys_coarse + eddy_flux_convergence)
        .assign_attrs(
            units="kg/kg/s",
            long_name="apparent moistening from high resolution data",
            description=(
                "Apparent moistening due to physics and sub-grid-scale advection. "
                "Given by "
                "sat adjustment (dycore) + physics tendency + eddy-flux-convergence"
            ),
        )
        .rename("Q2")
    )


def compute_fine_res_sources(
    data: FineResBudget, include_temperature_nudging: bool = False
) -> Tuple[xarray.DataArray, xarray.DataArray]:
    heating = apparent_heating(data, include_temperature_nudging)
    moistening = apparent_moistening(data)
    return heating, moistening


def _standardize_coords(
    ds: xarray.Dataset, time_shift=-timedelta(minutes=7, seconds=30)
) -> xarray.Dataset:
    ds_shifted = ds.assign(time=ds.time + time_shift)
    return gfdl_to_standard(ds_shifted).drop("tile")


def _open_fine_resolution_nudging_hybrid_dataset(
    # created by this commit:
    # https://github.com/VulcanClimateModeling/vcm-workflow-control/commit/3c852d0e4f8b86c4e88db9f29f0b8e484aeb77a1
    # I manually consolidated the metadata with zarr.consolidate_metadata
    fine_url: str = "gs://vcm-ml-experiments/default/2021-04-27/2020-05-27-40-day-X-SHiELD-simulation/fine-res-budget.zarr",  # noqa: E501
    # created by this commit
    # https://github.com/VulcanClimateModeling/vcm-workflow-control/commit/dd4498bcf3143d05095bf9ff4ca3f1341ba25330
    nudge_url="gs://vcm-ml-experiments/2021-04-13-n2f-c3072/3-hrly-ave-rad-precip-setting-30-min-rad-timestep-shifted-start-tke-edmf",  # noqa: E501
    include_temperature_nudging: bool = False,
) -> xarray.Dataset:

    fine = open_zarr_maybe_consolidated(fine_url)
    fine_shifted = _standardize_coords(fine)
    fine_shifted["Q1"], fine_shifted["Q2"] = compute_fine_res_sources(
        fine_shifted, include_temperature_nudging
    )

    return _open_nudged_hybrid_portion(fine_shifted, nudge_url)


def _open_precomputed_fine_resolution_nudging_hybrid_dataset(
    fine_url: str, nudge_url: str,
) -> xarray.Dataset:

    fine = open_zarr_maybe_consolidated(fine_url)

    return _open_nudged_hybrid_portion(fine, nudge_url)


class MLApparentSources(Protocol):
    dQ1: xarray.DataArray
    dQ2: xarray.DataArray


def _open_nudged_hybrid_portion(
    fine_shifted: xarray.Dataset, nudge_url: str
) -> MLApparentSources:

    nudge_physics_tendencies = open_zarr_maybe_consolidated(
        nudge_url + "/physics_tendencies.zarr",
    )
    nudge_state = open_zarr_maybe_consolidated(nudge_url + "/state_after_timestep.zarr")
    nudge_tends = open_zarr_maybe_consolidated(nudge_url + "/nudging_tendencies.zarr")

    merged = xarray.merge(
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

    return merged.astype(numpy.float32)


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


def _open_fine_resolution_dataset(
    fine_url: str, additional_dataset_urls: Optional[Sequence[str]]
) -> FineResBudget:

    fine = open_zarr_maybe_consolidated(fine_url)
    fine_shifted = _standardize_coords(fine)

    if additional_dataset_urls is not None:
        additional_datasets = []
        for url in additional_dataset_urls:
            additional_datasets.append(open_zarr_maybe_consolidated(url))
        merged = xarray.merge([fine_shifted, *additional_datasets], join="inner")
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
    merged: xarray.Dataset, approach: Approach, include_temperature_nudging: bool
) -> MLApparentSources:

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


def _add_nudging_tendencies(merged: xarray.Dataset):
    with xarray.set_options(keep_attrs=True):
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


def _add_dynamics_differences(merged: xarray.Dataset):
    with xarray.set_options(keep_attrs=True):
        Q1 = merged.Q1 + merged.air_temperature_tendency_due_to_dynamics
        Q2 = merged.Q2 + merged.specific_humidity_tendency_due_to_dynamics
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
    fine_source: xarray.DataArray, vertical_dim: str = "z"
) -> xarray.Dataset:
    if fine_source.sizes[vertical_dim] < 2:
        raise ValueError("vertical_dim must be greater than 1.")
    fine_source_new_bottom = fine_source.isel({vertical_dim: -2})
    fine_source_without_bottom = fine_source.isel({vertical_dim: slice(None, -1)})
    fine_source_extended_lower = xarray.concat(
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


def _ml_standard_names(merged: xarray.Dataset):

    # since ML target is Q1/Q2, dQ1=Q1 and pQ1=0 and same for moistening
    merged["dQ1"] = merged["Q1"]
    merged["dQ2"] = merged["Q2"]
    merged["pQ1"] = xarray.zeros_like(merged.Q1)
    merged["pQ1"].attrs = {"units": "K/s", "long_name": "coarse-res physics heating"}
    merged["pQ2"] = xarray.zeros_like(merged.Q2)
    merged["pQ2"].attrs = {
        "units": "kg/kg/s",
        "long_name": "coarse-res physics moistening",
    }

    return merged.astype(numpy.float32)


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
    budget: MLApparentSources = _compute_budget(
        merged, approach_enum, include_temperature_nudging=include_temperature_nudging
    )

    return XarrayMapper(budget)


def _open_precomputed_fine_resolution_dataset(
    fine_url: str, additional_dataset_urls: Optional[Sequence[str]] = None
) -> MLApparentSources:

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
