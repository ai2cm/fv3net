import numpy as np
import xarray as xr
from ..cubedsphere.constants import COORD_Z_CENTER
from ..types import Array
from .vertical_coordinates import pressure_at_interface, mass_integrate
from .constants import (
    _GRAVITY,
    _REFERENCE_SURFACE_PRESSURE,
    _POISSON_CONST,
    _RVGAS,
    _RDGAS,
    _LATENT_HEAT_FUSION,
    _LATENT_HEAT_VAPORIZATION_0_C,
    _SPECIFIC_ENTHALPY_LIQUID,
    _SPECIFIC_ENTHALPY_VAP0R,
    _FREEZING_TEMPERATURE,
    _DEFAULT_SURFACE_TEMPERATURE,
    _KG_M2_TO_MM,
    _KG_M2S_TO_MM_DAY,
    _SEC_PER_DAY,
    _SPECIFIC_HEAT_CONST_PRESSURE,
    _EARTH_RADIUS,
)


def potential_temperature(P: Array, T: Array) -> Array:
    return T * (_REFERENCE_SURFACE_PRESSURE / P) ** _POISSON_CONST


def hydrostatic_dz(
    T: xr.DataArray, q: xr.DataArray, delp: xr.DataArray, dim: str = COORD_Z_CENTER
) -> xr.DataArray:
    """Compute layer thickness assuming hydrostatic balance.

    Args:
        T: temperature
        q: specific humidity
        delp: pressure thickness
        dim (optional): name of vertical dimension. Defaults to "pfull".

    Returns:
        layer thicknesses dz
    """
    pi = pressure_at_interface(delp, dim_center=dim, dim_outer=dim)
    tv = T * (1 + (_RVGAS / _RDGAS - 1) * q)
    dlogp = np.log(pi).diff(dim)
    return -dlogp * _RDGAS * tv / _GRAVITY


def latent_heat_vaporization(T: Array) -> Array:
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
        latent_heat_flux: DataArray of latent heat flux in W/m2

    Returns:
        evaporation: DataArray of surface evaporation in mm/day
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
        specific_humidity: DataArray of specific humidity mixing ratio in kg/kg
        ice_water: DataArray of ice water mixing ratio in kg/kg
        liquid_water: DataArray of liquid water mixing ratio in kg/kg
        rain_water: DataArray of rain water mixing ratio in kg/kg
        snow_water: DataArray of snow water mixing ratio in kg/kg
        graupel_water: DataArray of graupel water mixing ratio in kg/kg
          
    Returns:
        total_water: DataArray of total water mixing ratio in kg/kg
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


def column_integrated_liquid_water_equivalent(
    specific_humidity: xr.DataArray, delp: xr.DataArray, vertical_dimension: str = "z"
) -> xr.DataArray:
    """Compute column-integrated liquid water equivalent from specific humidity
    
    Args:
        specific_humidity: DataArray of specific humidity in kg/kg
        delp: DataArray of pressure layer thicknesses in Pa
        vertical_dim: Name of vertical dimension; defaults to 'z'

    Returns:
        ci_liquid_water_equivalent: DataArray of water equivalent in mm
    """

    ci_liquid_water_equivalent = _KG_M2_TO_MM * mass_integrate(
        specific_humidity, delp, vertical_dimension
    )
    ci_liquid_water_equivalent = ci_liquid_water_equivalent.assign_attrs(
        {"long_name": "precipitable water", "units": "mm"}
    )

    return ci_liquid_water_equivalent


def liquid_ice_temperature(
    temperature: xr.DataArray,
    ice_water: xr.DataArray,
    liquid_water: xr.DataArray,
    rain_water: xr.DataArray,
    snow_water: xr.DataArray,
    graupel_water: xr.DataArray,
) -> xr.DataArray:
    """Compute liquid-ice temperature given temperature and hydrometeor mixing ratios,
    according to:
        T_LI = T - Lv ( ql + qr)  - (Lf + Lv)(qs + qg + qi),
    where Lv and Lf are latent heats of vaporization and fusion, respectively
    
    Args:
        temperature (T): DataArray of specific humidity mixing ratio in kg/kg
        ice_water (qi): DataArray of ice water mixing ratio in kg/kg
        liquid_water (ql): DataArray of liquid water mixing ratio in kg/kg
        rain_water (qr): DataArray of rain water mixing ratio in kg/kg
        snow_water (qs): DataArray of snow water mixing ratio in kg/kg
        graupel_water (qg): DataArray of graupel water mixing ratio in kg/kg
          
    Returns:
        liquid_ice_temperature (T_LI): DataArray of liquid-ice temperature in K
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


def column_integrated_heating_from_isobaric_transition(
    dtemperature_dt: xr.DataArray, delp: xr.DataArray, vertical_dim: str = "z"
) -> xr.DataArray:
    """Compute vertically-integrated heat tendencies assuming isobaric transition.
    
    Args:
        dtemperature_dt: DataArray of air temperature tendencies in K/s
        delp: DataArray of pressure layer thicknesses in Pa (NOT tendencies)
        vertical_dim: Name of vertical dimension; defaults to 'z'
          
    Returns:
        column_integrated_heating: DataArray of column total heat tendencies in W/m**2
    """

    column_integrated_heating = _SPECIFIC_HEAT_CONST_PRESSURE * mass_integrate(
        dtemperature_dt, delp, dim=vertical_dim
    )
    column_integrated_heating = column_integrated_heating.assign_attrs(
        {"long_name": "column integrated heating", "units": "W/m**2"}
    )

    return column_integrated_heating


def column_integrated_heating_from_isochoric_transition(
    dtemperature_dt: xr.DataArray, delp: xr.DataArray, vertical_dim: str = "z"
) -> xr.DataArray:
    """Compute vertically-integrated heat tendencies assuming isochoric transition.
    
    Args:
        dtemperature_dt: DataArray of air temperature tendencies in K/s
        delp: DataArray of pressure layer thicknesses in Pa (NOT tendencies)
        vertical_dim: Name of vertical dimension; defaults to 'z'
          
    Returns:
        column_integrated_heating: DataArray of column total heat tendencies in W/m**2
    """
    specific_heat = _SPECIFIC_HEAT_CONST_PRESSURE - _RDGAS
    column_integrated_heating = specific_heat * mass_integrate(
        dtemperature_dt, delp, dim=vertical_dim
    )
    column_integrated_heating = column_integrated_heating.assign_attrs(
        {"long_name": "column integrated heating", "units": "W/m**2"}
    )

    return column_integrated_heating


def minus_column_integrated_moistening(
    dsphum_dt: xr.DataArray, delp: xr.DataArray, vertical_dim: str = "z"
) -> xr.DataArray:
    """Compute negative of vertically-integrated moisture tendencies
    
    Args:
        dsphum_dt: DataArray of (positive) specific humidity tendencies in kg/kg/s
        delp: DataArray of pressure layer thicknesses in Pa (NOT tendencies)
        vertical_dim: Name of vertical dimension; defaults to 'z'
          
    Returns:
        column_integrated_heating: DataArray of negative of column total moisture
        tendencies in mm/day
    """

    minus_col_int_moistening = _KG_M2S_TO_MM_DAY * mass_integrate(
        dsphum_dt * -1, delp, dim=vertical_dim
    )
    minus_col_int_moistening = minus_col_int_moistening.assign_attrs(
        {"long_name": "negative of column integrated moistening", "units": "mm/day"}
    )

    return minus_col_int_moistening


def mass_streamfunction(
    northward_wind: xr.DataArray, lat: str = "latitude", plev: str = "pressure",
) -> xr.DataArray:
    """Compute overturning streamfunction from northward wind on pressure grid.

    Args:
        northward_wind: the meridional wind which depends on pressure and latitude.
            May have additional dimensions.
        lat: name of latitude dimension. Latitude must be in degrees.
        plev: name of pressure dimension. Pressure must be in Pa.

    Returns:
        mass streamfunction on same coordinates as northward_wind
    """
    pressure_thickness = northward_wind[plev].diff(plev, label="lower")
    psi = (northward_wind * pressure_thickness).cumsum(dim=plev)
    psi = 2 * np.pi * _EARTH_RADIUS * np.cos(np.deg2rad(psi[lat])) * psi / _GRAVITY
    # extend psi assuming it is constant so it has same pressure coordinate as input
    bottom_level_pressure = northward_wind[plev].isel({plev: -1})
    bottom_level_psi = psi.isel({plev: -1}).assign_coords({plev: bottom_level_pressure})
    psi = xr.concat([psi, bottom_level_psi], dim=plev)
    return (psi / 1e9).assign_attrs(long_name="mass streamfunction", units="Gkg/s")


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
