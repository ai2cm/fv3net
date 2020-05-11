import numpy as np
import xarray as xr
from ..cubedsphere.constants import COORD_Z_CENTER, COORD_Z_OUTER
from .calc import mass_integrate


# following are defined as in FV3GFS model (see FV3/fms/constants/constants.f90)
_GRAVITY = 9.80665  # m /s2
_RDGAS = 287.05  # J / K / kg
_RVGAS = 461.5  # J / K / kg
_LATENT_HEAT_VAPORIZATION_0_C = 2.5e6  # J/kg
_LATENT_HEAT_FUSION = 3.3358e5  # J/kg
_SPECIFIC_ENTHALPY_LIQUID = 4185.5
_SPECIFIC_ENTHALPY_VAP0R = 1846
_SPECIFIC_HEAT_CONST_PRESSURE = 1004
_FREEZING_TEMPERATURE = 273.15
_POISSON_CONST = 0.2854

_DEFAULT_SURFACE_TEMPERATURE = _FREEZING_TEMPERATURE + 15

_TOA_PRESSURE = 300.0  # Pa
_REFERENCE_SURFACE_PRESSURE = 100000  # reference pressure for potential temp [Pa]
_REVERSE = slice(None, None, -1)

_SEC_PER_DAY = 86400
_KG_M2S_TO_MM_DAY = (1e3 * 86400) / 997.0
_KG_M2_TO_MM = 1000.0 / 997


def potential_temperature(P, T):
    return T * (_REFERENCE_SURFACE_PRESSURE / P) ** _POISSON_CONST


def pressure_at_interface(
    delp, toa_pressure=_TOA_PRESSURE, dim_center=COORD_Z_CENTER, dim_outer=COORD_Z_OUTER
):
    """ Compute pressure at layer interfaces

    Args:
        delp (xr.DataArray): pressure thicknesses
        toa_pressure (float, optional): pressure at the top of atmosphere.
            Defaults to 300Pa.
        dim_center (str, optional): name of vertical dimension for delp.
            Defaults to "pfull".
        dim_outer (str, optional): name of vertical dimension for output pressure.
            Defaults to "phalf".

    Returns:
        xr.DataArray: atmospheric pressure at layer interfaces
    """
    top = xr.full_like(delp.isel({dim_center: [0]}), toa_pressure).variable
    delp_with_top = top.concat([top, delp.variable], dim=dim_center)
    pressure = delp_with_top.cumsum(dim_center)
    return _add_coords_to_interface_variable(
        pressure, delp, dim_center=dim_center
    ).rename({dim_center: dim_outer})


def height_at_interface(dz, phis, dim_center=COORD_Z_CENTER, dim_outer=COORD_Z_OUTER):
    """ Compute geopotential height at layer interfaces

    Args:
        dz (xr.DataArray): layer height thicknesses
        phis (xr.DataArray): surface geopotential
        dim_center (str, optional): name of vertical dimension for dz.
            Defaults to "pfull".
        dim_outer (str, optional): name of vertical dimension for output heights.
            Defaults to "phalf".

    Returns:
        xr.DataArray: height at layer interfaces
    """
    bottom = phis.broadcast_like(dz.isel({dim_center: [0]})) / _GRAVITY
    dzv = -dz.variable  # dz is negative in model
    dz_with_bottom = dzv.concat([dzv, bottom], dim=dim_center)
    height = (
        dz_with_bottom.isel({dim_center: _REVERSE})
        .cumsum(dim_center)
        .isel({dim_center: _REVERSE})
    )
    return _add_coords_to_interface_variable(height, dz, dim_center=dim_center).rename(
        {dim_center: dim_outer}
    )


def _add_coords_to_interface_variable(
    dv_outer: xr.Variable, da_center: xr.DataArray, dim_center: str = COORD_Z_CENTER
):
    """Assign all coords except vertical from da_center to dv_outer """
    if dim_center in da_center.coords:
        return xr.DataArray(dv_outer, coords=da_center.drop(dim_center).coords)
    else:
        return xr.DataArray(dv_outer, coords=da_center.coords)


def pressure_at_midpoint(delp, toa_pressure=_TOA_PRESSURE, dim=COORD_Z_CENTER):
    """ Compute pressure at layer midpoints by linear interpolation

    Args:
        delp (xr.DataArray): pressure thicknesses
        toa_pressure (float, optional): pressure at the top of atmosphere.
            Defaults to 300Pa.
        dim (str, optional): name of vertical dimension for delp. Defaults to "pfull".

    Returns:
        xr.DataArray: atmospheric pressure at layer midpoints
    """
    pi = pressure_at_interface(delp, toa_pressure=toa_pressure, dim_center=dim)
    return _interface_to_midpoint(pi, dim_center=dim)


def height_at_midpoint(dz, phis, dim=COORD_Z_CENTER):
    """ Compute geopotential height at layer midpoints by linear interpolation

    Args:
        dz (xr.DataArray): layer height thicknesses
        phis (xr.DataArray): surface geopotential
        dim (str, optional): name of vertical dimension for dz. Defaults to "pfull".

    Returns:
        xr.DataArray: height at layer midpoints
    """
    zi = height_at_interface(dz, phis, dim_center=dim)
    return _interface_to_midpoint(zi, dim_center=dim)


def _interface_to_midpoint(da, dim_center=COORD_Z_CENTER, dim_outer=COORD_Z_OUTER):
    da_mid = (
        da.isel({dim_outer: slice(0, -1)}) + da.isel({dim_outer: slice(1, None)})
    ) / 2
    return da_mid.rename({dim_outer: dim_center})


def pressure_at_midpoint_log(delp, toa_pressure=_TOA_PRESSURE, dim=COORD_Z_CENTER):
    """ Compute pressure at layer midpoints following Eq. 3.17 of Simmons
    and Burridge (1981), MWR.

    Args:
        delp (xr.DataArray): pressure thicknesses
        toa_pressure (float, optional): pressure at the top of atmosphere.
            Defaults to 300Pa.
        dim (str, optional): name of vertical dimension for delp. Defaults to "pfull".

    Returns:
        xr.DataArray: atmospheric pressure at layer midpoints
    """
    pi = pressure_at_interface(
        delp, toa_pressure=toa_pressure, dim_center=dim, dim_outer=dim
    )
    dlogp = np.log(pi).diff(dim)
    return delp / dlogp


def hydrostatic_dz(T, q, delp, dim=COORD_Z_CENTER):
    """ Compute layer thickness assuming hydrostatic balance

    Args:
        T (xr.DataArray): temperature
        q (xr.DataArray): specific humidity
        delp (xr.DataArray): pressure thickness
        dim (str, optional): name of vertical dimension. Defaults to "pfull".

    Returns:
        xr.DataArray: layer thicknesses dz
    """
    pi = pressure_at_interface(delp, dim_center=dim, dim_outer=dim)
    tv = T * (1 + (_RVGAS / _RDGAS - 1) * q)
    dlogp = np.log(pi).diff(dim)
    return -dlogp * _RDGAS * tv / _GRAVITY


def dz_and_top_to_phis(top_height, dz, dim=COORD_Z_CENTER):
    """ Compute surface geopotential from model top height and layer thicknesses"""
    return _GRAVITY * (top_height + dz.sum(dim=dim))


def latent_heat_vaporization(T):
    return _LATENT_HEAT_VAPORIZATION_0_C + (
        _SPECIFIC_ENTHALPY_LIQUID - _SPECIFIC_ENTHALPY_VAP0R
    ) * (T - _FREEZING_TEMPERATURE)


def net_heating(
    dlw_sfc,
    dsw_sfc,
    ulw_sfc,
    ulw_toa,
    usw_sfc,
    usw_toa,
    dsw_toa,
    shf,
    surface_rain_rate,
    surface_temperature=_FREEZING_TEMPERATURE + 10,
):
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
    lhf, surface_temperature=_DEFAULT_SURFACE_TEMPERATURE
):
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


def net_precipitation(lhf, prate):
    da = (prate - latent_heat_flux_to_evaporation(lhf)) * _SEC_PER_DAY
    da.attrs = {"long_name": "net precipitation from model physics", "units": "mm/day"}
    return da


def surface_pressure_from_delp(
    delp: xr.DataArray, p_toa: float = 300.0, vertical_dim: str = "z"
) -> xr.DataArray:
    """Compute surface pressure from delp in Pa
    
    Args:
        delp: DataArray of pressure layer thicknesses in Pa
        p_toa: Top of atmosphere pressure in Pa; optional, defaults to 300
        vertical_dim: Name of vertical dimension; defaults to 'z'
        
    Returns:
        surface_pressure: DataArray of surface pressure in Pa
    """

    surface_pressure = delp.sum(dim=vertical_dim) + p_toa
    surface_pressure = surface_pressure.assign_attrs(
        {"long_name": "surface pressure", "units": "Pa"}
    )

    return surface_pressure


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

    ci_liquid_water_equivalent = _KG_M2_TO_MM * (
        (delp / _GRAVITY) * specific_humidity
    ).sum(vertical_dimension)
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


def column_integrated_heat(
    temperature: xr.DataArray, delp: xr.DataArray, vertical_dim: str = "z"
) -> xr.DataArray:
    """Compute vertically-integrated total heat
    
    Args:
        temperature: DataArray of air temperature in K
        delp: DataArray of pressure layer thicknesses in Pa
        vertical_dim: Name of vertical dimension; defaults to 'z'
          
    Returns:
        column_integrated_heat: DataArray of column total heat in J/m**2
    """

    column_integrated_heat = (
        _SPECIFIC_HEAT_CONST_PRESSURE * (delp / _GRAVITY) * temperature
    ).sum(vertical_dim)
    column_integrated_heat = column_integrated_heat.assign_attrs(
        {"long_name": "column integrated heat", "units": "J/m**2"}
    )

    return column_integrated_heat


def column_integrated_heating(
    dtemperature_dt: xr.DataArray, delp: xr.DataArray, vertical_dim: str = "z"
) -> xr.DataArray:
    """Compute vertically-integrated heat tendencies
    
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
        -dsphum_dt, delp, dim=vertical_dim
    )
    minus_col_int_moistening = minus_col_int_moistening.assign_attrs(
        {"long_name": "negative of column integrated moistening", "units": "mm/day"}
    )

    return minus_col_int_moistening
