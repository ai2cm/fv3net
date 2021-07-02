from typing import Mapping

import xarray as xr

import vcm

from .registry import Registry


def merge_derived(diags_dict: Mapping[str, xr.DataArray]) -> xr.Dataset:
    out = xr.Dataset()
    for name, da in diags_dict.items():
        out[name] = da
    return out


derived_registry = Registry(merge_derived)


@derived_registry.register("mass_streamfunction_pressure_level_zonal_time_mean")
def psi_value(diags: xr.Dataset) -> xr.DataArray:
    northward_wind = diags["northward_wind_pressure_level_zonal_time_mean"]
    return vcm.mass_streamfunction(northward_wind)


@derived_registry.register("mass_streamfunction_pressure_level_zonal_bias")
def psi_bias(diags: xr.Dataset) -> xr.DataArray:
    northward_wind_bias = diags["northward_wind_pressure_level_zonal_bias"]
    return vcm.mass_streamfunction(northward_wind_bias)


@derived_registry.register("mass_streamfunction_300_700_zonal_and_time_mean")
def psi_value_mid_troposphere(diags: xr.Dataset) -> xr.DataArray:
    northward_wind = diags["northward_wind_pressure_level_zonal_time_mean"]
    psi = vcm.mass_streamfunction(northward_wind).sel(pressure=slice(30000, 70000))
    psi_mid_trop = psi.weighted(psi.pressure).mean("pressure")
    return psi_mid_trop.assign_attrs(
        long_name="mass streamfunction 300-700hPa average", units="Gkg/s"
    )


@derived_registry.register("mass_streamfunction_300_700_zonal_bias")
def psi_bias_mid_troposphere(diags: xr.Dataset) -> xr.DataArray:
    northward_wind_bias = diags["northward_wind_pressure_level_zonal_bias"]
    psi = vcm.mass_streamfunction(northward_wind_bias).sel(pressure=slice(30000, 70000))
    psi_mid_trop = psi.weighted(psi.pressure).mean("pressure")
    return psi_mid_trop.assign_attrs(
        long_name="mass streamfunction 300-700hPa average", units="Gkg/s"
    )
