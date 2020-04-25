from vcm.constants import SPECIFIC_HEAT_CONST_PRESSURE, GRAVITY, kg_m2s_to_mm_day
from vcm.calc import mass_integrate
from vcm.calc.thermo import (
    latent_heat_vaporization,
    _SPECIFIC_HEAT_CONST_PRESSURE,
    latent_heat_flux_to_evaporation,
)
import xarray as xr

_LATENT_HEAT_FUSION = 3.34e5


def psurf_from_delp(delp: xr.DataArray, p_toa: float = 300.0) -> xr.DataArray:
    """Compute surface pressure from delp
    """
    psurf = delp.sum(dim="z") + p_toa
    psurf = psurf.assign_attrs({"long_name": "surface pressure", "units": "Pa"})
    return psurf


def total_water(
    sphum: xr.DataArray,
    ice_wat: xr.DataArray,
    liq_wat: xr.DataArray,
    rainwat: xr.DataArray,
    snowwat: xr.DataArray,
    graupel: xr.DataArray,
) -> xr.DataArray:
    """Compute total water species mixing ratio
    """
    total_water = sphum + ice_wat + liq_wat + rainwat + snowwat + graupel
    total_water = total_water.assign_attrs(
        {"long_name": "total water species mixing ratio", "units": "kg/kg"}
    )
    return total_water


def precipitable_water(sphum: xr.DataArray, delp: xr.DataArray) -> xr.DataArray:
    """Compute vertically-integrated precipitable water
    """
    pw = ((delp / GRAVITY) * sphum).sum("z")
    pw = pw.assign_attrs({"long_name": "precipitable water", "units": "mm"})
    return pw


def liquid_ice_temperature(
    T: xr.DataArray,
    ice_wat: xr.DataArray,
    liq_wat: xr.DataArray,
    rainwat: xr.DataArray,
    snowwat: xr.DataArray,
    graupel: xr.DataArray,
) -> xr.DataArray:

    liquid_ice_temperature = (
        T
        - (latent_heat_vaporization(T) / _SPECIFIC_HEAT_CONST_PRESSURE)
        * (rainwat + liq_wat)
        - (
            (latent_heat_vaporization(T) + _LATENT_HEAT_FUSION)
            / _SPECIFIC_HEAT_CONST_PRESSURE
        )
        * (ice_wat + snowwat + graupel)
    )

    liquid_ice_temperature = liquid_ice_temperature.assign_attrs(
        {"long_name": "liquid ice temperature", "units": "K"}
    )

    return liquid_ice_temperature


def total_heat(T: xr.DataArray, delp: xr.DataArray) -> xr.DataArray:
    """Compute vertically-integrated total heat
    """
    total_heat = (SPECIFIC_HEAT_CONST_PRESSURE * (delp / GRAVITY) * T).sum("z")
    total_heat = total_heat.assign_attrs(
        {"long_name": "column integrated heat", "units": "J/m**2"}
    )
    return total_heat


def column_integrated_heating(dT_dt: xr.DataArray, delp: xr.DataArray) -> xr.DataArray:
    """Compute vertically-integrated heat tendencies
    """
    dT_dt_integrated = SPECIFIC_HEAT_CONST_PRESSURE * mass_integrate(
        dT_dt, delp, dim="z"
    )
    dT_dt_integrated = dT_dt_integrated.assign_attrs(
        {"long_name": "column integrated heating", "units": "W/m^2"}
    )
    return dT_dt_integrated


def minus_column_integrated_moistening(
    minus_dsphum_dt: xr.DataArray, delp: xr.DataArray
) -> xr.DataArray:
    """Compute negative of vertically-integrated moisture tendencies
    """
    minus_dsphum_dt_integrated = kg_m2s_to_mm_day * mass_integrate(
        minus_dsphum_dt, delp, dim="z"
    )
    minus_dsphum_dt_integrated = minus_dsphum_dt_integrated.assign_attrs(
        {"long_name": "negative column integrated moistening", "units": "mm/day"}
    )
    return minus_dsphum_dt_integrated


def evaporation(latent_heat_flux: xr.DataArray) -> xr.DataArray:
    """Compute surface evaporation in mm/day from latent heat flux"""

    evaporation = kg_m2s_to_mm_day * latent_heat_flux_to_evaporation(latent_heat_flux)
    evaporation = evaporation.assign_attrs(
        {"long_name": "surface evaporation", "units": "mm/day"}
    )

    return evaporation
