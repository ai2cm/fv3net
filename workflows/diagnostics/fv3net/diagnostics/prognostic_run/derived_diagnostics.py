from typing import Sequence, Tuple

import xarray as xr

import vcm

from fv3net.diagnostics._shared.registry import Registry
from .constants import WVP, COL_DRYING


def merge_derived(diags: Sequence[Tuple[str, xr.DataArray]]) -> xr.Dataset:
    out = xr.Dataset()
    for name, da in diags:
        if len(da.dims) != 0:
            # don't add empty DataArrays, which are returned in case of missing inputs
            out[name] = da
    return out


# all functions added to this registry must take a single xarray Dataset as
# input and return a single xarray DataArray
derived_registry = Registry(merge_derived)


@derived_registry.register("mass_streamfunction_pressure_level_zonal_time_mean")
def psi_value(diags: xr.Dataset) -> xr.DataArray:
    if "northward_wind_pressure_level_zonal_time_mean" not in diags:
        return xr.DataArray()
    northward_wind = diags["northward_wind_pressure_level_zonal_time_mean"]
    return vcm.mass_streamfunction(northward_wind)


@derived_registry.register("mass_streamfunction_pressure_level_zonal_bias")
def psi_bias(diags: xr.Dataset) -> xr.DataArray:
    if "northward_wind_pressure_level_zonal_bias" not in diags:
        return xr.DataArray()
    northward_wind_bias = diags["northward_wind_pressure_level_zonal_bias"]
    return vcm.mass_streamfunction(northward_wind_bias)


@derived_registry.register("mass_streamfunction_300_700_zonal_and_time_mean")
def psi_value_mid_troposphere(diags: xr.Dataset) -> xr.DataArray:
    if "northward_wind_pressure_level_zonal_time_mean" not in diags:
        return xr.DataArray()
    northward_wind = diags["northward_wind_pressure_level_zonal_time_mean"]
    psi = vcm.mass_streamfunction(northward_wind).sel(pressure=slice(30000, 70000))
    psi_mid_trop = psi.weighted(psi.pressure).mean("pressure")
    return psi_mid_trop.assign_attrs(
        long_name="mass streamfunction 300-700hPa average", units="Gkg/s"
    )


@derived_registry.register("mass_streamfunction_300_700_zonal_bias")
def psi_bias_mid_troposphere(diags: xr.Dataset) -> xr.DataArray:
    if "northward_wind_pressure_level_zonal_bias" not in diags:
        return xr.DataArray()
    northward_wind_bias = diags["northward_wind_pressure_level_zonal_bias"]
    psi = vcm.mass_streamfunction(northward_wind_bias).sel(pressure=slice(30000, 70000))
    psi_mid_trop = psi.weighted(psi.pressure).mean("pressure")
    return psi_mid_trop.assign_attrs(
        long_name="mass streamfunction 300-700hPa average", units="Gkg/s"
    )


@derived_registry.register(f"conditional_average_of_{COL_DRYING}_on_{WVP}")
def conditional_average(diags: xr.Dataset) -> xr.DataArray:
    count_name = f"{WVP}_versus_{COL_DRYING}_hist_2d"
    q2_bin_name = f"{COL_DRYING}_bins"
    q2_bin_widths_name = f"{COL_DRYING}_bin_width_hist_2d"
    if count_name not in diags:
        return xr.DataArray()
    count = diags[count_name]
    q2_bin_centers = diags[q2_bin_name] + 0.5 * diags[q2_bin_widths_name]
    average = vcm.weighted_average(q2_bin_centers, count, dims=[q2_bin_name])
    return average.assign_attrs(
        long_name="Conditional average of -<Q2> on water vapor path", units="mm/day"
    )
