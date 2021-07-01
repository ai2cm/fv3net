from toolz.functoolz import curry
from typing import Callable

import xarray as xr

import vcm

_DERIVED_DIAGNOSTICS = []


@curry
def add_to_derived_diagnostics(
    diagnostic_name: str, func: Callable[[xr.Dataset], xr.DataArray]
):
    """Register a function to be used for computing a derived diagnostics

    This function will be passed the diagnostics xarray dataset,
    and should return a DataArray.
    """

    def myfunc(diags):
        derived_diag = func(diags)
        return derived_diag.rename(diagnostic_name)

    _DERIVED_DIAGNOSTICS.append(myfunc)
    return func


def add_derived_diagnostics(diags: xr.Dataset) -> xr.Dataset:
    derived_diags = {}
    for diag_func in _DERIVED_DIAGNOSTICS:
        da = diag_func(diags)
        derived_diags[da.name] = da
    diags = diags.merge(derived_diags)
    return diags


@add_to_derived_diagnostics("mass_streamfunction_pressure_level_zonal_time_mean")
def psi_value(diags: xr.Dataset) -> xr.DataArray:
    northward_wind = diags["northward_wind_pressure_level_zonal_time_mean"]
    return vcm.mass_streamfunction(northward_wind)


@add_to_derived_diagnostics("mass_streamfunction_pressure_level_zonal_bias")
def psi_bias(diags: xr.Dataset) -> xr.DataArray:
    northward_wind_bias = diags["northward_wind_pressure_level_zonal_bias"]
    return vcm.mass_streamfunction(northward_wind_bias)


@add_to_derived_diagnostics("mass_streamfunction_300_700_zonal_and_time_mean")
def psi_value_mid_troposphere(diags: xr.Dataset) -> xr.DataArray:
    northward_wind = diags["northward_wind_pressure_level_zonal_time_mean"]
    psi = vcm.mass_streamfunction(northward_wind).sel(pressure=slice(30000, 70000))
    psi_mid_trop = psi.weighted(psi.pressure).mean("pressure")
    return psi_mid_trop.assign_attrs(
        long_name="mass streamfunction 300-700hPa average", units="Gkg/s"
    )


@add_to_derived_diagnostics("mass_streamfunction_300_700_zonal_bias")
def psi_bias_mid_troposphere(diags: xr.Dataset) -> xr.DataArray:
    northward_wind_bias = diags["northward_wind_pressure_level_zonal_bias"]
    psi = vcm.mass_streamfunction(northward_wind_bias).sel(pressure=slice(30000, 70000))
    psi_mid_trop = psi.weighted(psi.pressure).mean("pressure")
    return psi_mid_trop.assign_attrs(
        long_name="mass streamfunction 300-700hPa average", units="Gkg/s"
    )

