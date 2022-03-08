from typing import Union
import numpy as np
import xarray as xr
from .constants import (
    _GRAVITY,
    _REFERENCE_SURFACE_PRESSURE,
    _POISSON_CONST,
    _RDGAS,
    _RVGAS,
    _LATENT_HEAT_FUSION,
    _LATENT_HEAT_VAPORIZATION_0_C,
    _SPECIFIC_ENTHALPY_LIQUID,
    _SPECIFIC_ENTHALPY_VAP0R,
    _FREEZING_TEMPERATURE,
    _DEFAULT_SURFACE_TEMPERATURE,
    _KG_M2S_TO_MM_DAY,
    _SEC_PER_DAY,
    _SPECIFIC_HEAT_CONST_PRESSURE,
)


def potential_temperature(P, T):
    return T * (_REFERENCE_SURFACE_PRESSURE / P) ** _POISSON_CONST


def latent_heat_vaporization(T):
    return _LATENT_HEAT_VAPORIZATION_0_C + (
        _SPECIFIC_ENTHALPY_LIQUID - _SPECIFIC_ENTHALPY_VAP0R
    ) * (T - _FREEZING_TEMPERATURE)


def net_heating(
    dlw_sfc: xr.DataArray,
    dsw_sfc: xr.DataArray,
    ulw_sfc: xr.DataArray,
    ulw_toa: xr.DataArray,
    usw_sfc: xr.DataArray,
    usw_toa: xr.DataArray,
    dsw_toa: xr.DataArray,
    shf: xr.DataArray,
    surface_rain_rate: xr.DataArray,
    surface_temperature: float = _FREEZING_TEMPERATURE + 10,
) -> xr.DataArray:
    """A dataarray implementation of ``net_heating_from_dataset``

    All argument except for ``cond_int`` should be in W/m2. cond_int is in units
    kg/kg/s which is equivalent to a surface precipitation rate in mm/s. All
    precipitation is assumed to fall as liquid water.

    If provided, the surface temperature in K will be used to calculate the latent heat
    of vaporization with the FV3 formulas.
    """

    lv = latent_heat_vaporization(surface_temperature)
    da = (
        -dlw_sfc
        - dsw_sfc
        + ulw_sfc
        - ulw_toa
        + usw_sfc
        - usw_toa
        + dsw_toa
        + shf
        + surface_rain_rate * lv
    )
    da.attrs = {"long_name": "net heating from model physics", "units": "W/m^2"}
    return da


def latent_heat_flux_to_evaporation(
    lhf: xr.DataArray, surface_temperature: float = _DEFAULT_SURFACE_TEMPERATURE
) -> xr.DataArray:
    """Compute evaporation from latent heat flux

    Args:
        lhf: W/m2
        surface_temperature: degrees K

    Returns:
        evaporation: kg/s/m^2
    """

    return lhf / latent_heat_vaporization(surface_temperature)


def surface_evaporation_mm_day_from_latent_heat_flux(
    latent_heat_flux: xr.DataArray,
) -> xr.DataArray:
    """Compute surface evaporation in mm/day from latent heat flux

    Args:
        latent_heat_flux: latent heat flux in W/m2

    Returns:
        evaporation: surface evaporation in mm/day
    """

    surface_evaporation = _KG_M2S_TO_MM_DAY * latent_heat_flux_to_evaporation(
        latent_heat_flux
    )
    surface_evaporation = surface_evaporation.assign_attrs(
        {"long_name": "surface evaporation", "units": "mm/day"}
    )

    return surface_evaporation


def net_precipitation(lhf: xr.DataArray, prate: xr.DataArray) -> xr.DataArray:
    da = (prate - latent_heat_flux_to_evaporation(lhf)) * _SEC_PER_DAY
    da.attrs = {"long_name": "net precipitation from model physics", "units": "mm/day"}
    return da


def total_water(
    specific_humidity: xr.DataArray,
    ice_water: xr.DataArray,
    liquid_water: xr.DataArray,
    rain_water: xr.DataArray,
    snow_water: xr.DataArray,
    graupel_water: xr.DataArray,
) -> xr.DataArray:
    """Compute total water species mixing ratio

    Args:
        specific_humidity: specific humidity mixing ratio in kg/kg
        ice_water: ice water mixing ratio in kg/kg
        liquid_water: liquid water mixing ratio in kg/kg
        rain_water: rain water mixing ratio in kg/kg
        snow_water: snow water mixing ratio in kg/kg
        graupel_water: graupel water mixing ratio in kg/kg

    Returns:
        total_water: total water mixing ratio in kg/kg
    """

    total_water = (
        specific_humidity
        + ice_water
        + liquid_water
        + rain_water
        + snow_water
        + graupel_water
    )
    total_water = total_water.assign_attrs(
        {"long_name": "total water species mixing ratio", "units": "kg/kg"}
    )

    return total_water


def liquid_ice_temperature(
    temperature: xr.DataArray,
    ice_water: xr.DataArray,
    liquid_water: xr.DataArray,
    rain_water: xr.DataArray,
    snow_water: xr.DataArray,
    graupel_water: xr.DataArray,
) -> xr.DataArray:
    """Compute liquid-ice temperature given temperature and hydrometeor mixing ratios,
    according to::

        T_LI = T - Lv ( ql + qr)  - (Lf + Lv)(qs + qg + qi),

    where Lv and Lf are latent heats of vaporization and fusion, respectively

    Args:
        temperature (T): specific humidity mixing ratio in kg/kg
        ice_water (qi): ice water mixing ratio in kg/kg
        liquid_water (ql): liquid water mixing ratio in kg/kg
        rain_water (qr): rain water mixing ratio in kg/kg
        snow_water (qs): snow water mixing ratio in kg/kg
        graupel_water (qg): graupel water mixing ratio in kg/kg

    Returns:
        liquid_ice_temperature (T_LI): liquid-ice temperature in K
    """

    liquid_adjustment = (
        latent_heat_vaporization(temperature) / _SPECIFIC_HEAT_CONST_PRESSURE
    ) * (rain_water + liquid_water)

    ice_adjustment = (
        (latent_heat_vaporization(temperature) + _LATENT_HEAT_FUSION)
        / _SPECIFIC_HEAT_CONST_PRESSURE
    ) * (ice_water + snow_water + graupel_water)

    liquid_ice_temperature = temperature - liquid_adjustment - ice_adjustment

    liquid_ice_temperature = liquid_ice_temperature.assign_attrs(
        {"long_name": "liquid-ice temperature", "units": "K"}
    )

    return liquid_ice_temperature


def internal_energy(temperature: xr.DataArray) -> xr.DataArray:
    """Compute internal energy from temperature.

    Args:
        temperature: in units of K.

    Returns:
        c_v * temperature in units of J/kg.
    """
    internal_energy = (_SPECIFIC_HEAT_CONST_PRESSURE - _RDGAS) * temperature
    internal_energy = internal_energy.assign_attrs(
        {"long_name": "internal energy", "units": "J/kg"}
    )
    return internal_energy


def saturation_pressure(temperature, math=np):
    """Saturation vapor pressure from temperature using `August Roche Magnus`_ formula.

    Args:
        temperature: temperature in units of K.
        math (optional): module with an exponential function `exp`. Defaults to numpy.

    Returns:
        Saturation vapor pressure.

    .. _August Roche Magnus:
        https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation#Meteorology_and_climatology # noqa
    """
    temperature_celsius = temperature - 273.15
    return 610.94 * math.exp(
        17.625 * temperature_celsius / (temperature_celsius + 243.04)
    )


def relative_humidity(temperature, specific_humidity, density, math=np):
    """Relative humidity from temperature, specific humidity and density.

    Args:
        temperature: air temperature in units of K.
        specific_humidity: in units of kg/kg.
        density: air density in kg/m**3.
        math (optional): module with an exponential function `exp`. Defaults to numpy.

    Returns:
        Relative humidity.
    """
    partial_pressure = _RVGAS * specific_humidity * density * temperature
    return partial_pressure / saturation_pressure(temperature, math=math)


def specific_humidity_from_rh(temperature, relative_humidity, density, math=np):
    """Specific humidity from temperature, relative humidity and density.

    Args:
        temperature: air temperature in units of K.
        relative_humidity: unitless.
        density: air density in kg/m**3.
        math (optional): module with an exponential function `exp`. Defaults to numpy.

    Returns:
        Specific humidity in kg/kg.
    """
    es = saturation_pressure(temperature, math=math)
    partial_pressure = relative_humidity * es

    return partial_pressure / _RVGAS / density / temperature


def density(delp, delz, math=np):
    """Compute density from pressure and vertical thickness.

    Args:
        delp: pressure thickness of atmospheric layer in units of Pa.
        delz: height of atmospheric layer in units of m.
        math (optional): module with an absolute function `abs`. Defaults to numpy.

    Returns:
        Density in units of kg/m**3.
    """
    return math.abs(delp / delz / _GRAVITY)


def pressure_thickness(density, delz, math=np):
    """Compute density from pressure and vertical thickness.

    Args:
        density: density of atmospheric layer in units of kg/m**3.
        delz: height of atmospheric layer in units of m.
        math (optional): module with an absolute function `abs`. Defaults to numpy.

    Returns:
        Pressure thickness in units of Pa.
    """
    return math.abs(density * delz * _GRAVITY)


def layer_mass(delp):
    """Layer mass in kg/m^2 from ``delp`` in Pa"""
    return delp / _GRAVITY


def moist_static_energy_tendency(
    temperature_tendency: xr.DataArray,
    specific_humidity_tendency: xr.DataArray,
    temperature: Union[float, xr.DataArray] = _FREEZING_TEMPERATURE,
) -> xr.DataArray:
    """Compute moist static energy tendency from temperature and humidity tendencies.

    Args:
        temperature_tendency: in units of K/s
        specific_humidity_tendency: in units of kg/kg/s
        temperature: in units of K, for computing latent heat of vaporization
    """
    mse_tendency = (
        (_SPECIFIC_HEAT_CONST_PRESSURE - _RDGAS) * temperature_tendency
        + latent_heat_vaporization(temperature) * specific_humidity_tendency
    )
    return mse_tendency.assign_attrs(
        units="W/kg", long_name="tendency of moist static energy"
    )


def temperature_tendency(
    moist_static_energy_tendency: xr.DataArray,
    specific_humidity_tendency: xr.DataArray,
    temperature: Union[float, xr.DataArray] = _FREEZING_TEMPERATURE,
) -> xr.DataArray:
    """Compute temperature tendency from moist static energy and humidity tendencies.

    Args:
        moist static energy_tendency: in units of W/kg
        specific_humidity_tendency: in units of kg/kg/s
        temperature: in units of K, for computing latent heat of vaporization
    """
    heat_capacity = _SPECIFIC_HEAT_CONST_PRESSURE - _RDGAS
    temperature_tendency = (
        moist_static_energy_tendency
        - latent_heat_vaporization(temperature) * specific_humidity_tendency
    ) / heat_capacity
    return temperature_tendency.assign_attrs(
        units="K/s", long_name="tendency of air temperature"
    )
