import numpy as np
import xarray as xr
from ...cubedsphere.constants import COORD_Z_CENTER, COORD_Z_OUTER
from .constants import (
    _GRAVITY,
    _RVGAS,
    _RDGAS,
    _KG_M2_TO_MM,
    _EARTH_RADIUS,
    _SPECIFIC_HEAT_CONST_PRESSURE,
    _KG_M2S_TO_MM_DAY,
)

_TOA_PRESSURE = 300.0  # Pa
_REVERSE = slice(None, None, -1)


def mass_integrate(
    da: xr.DataArray, delp: xr.DataArray, dim: str = COORD_Z_CENTER
) -> xr.DataArray:
    """Compute mass-weighted vertical integral."""
    return (da * delp / _GRAVITY).sum(dim)


def pressure_at_interface(
    delp: xr.DataArray,
    toa_pressure: float = _TOA_PRESSURE,
    dim_center: str = COORD_Z_CENTER,
    dim_outer: str = COORD_Z_OUTER,
) -> xr.DataArray:
    """Compute pressure at layer interfaces.

    Args:
        delp: pressure thicknesses
        toa_pressure (optional): pressure at the top of atmosphere.
            Defaults to 300Pa.
        dim_center (optional): name of vertical dimension for delp.
            Defaults to "pfull".
        dim_outer (optional): name of vertical dimension for output pressure.
            Defaults to "phalf".

    Returns:
        atmospheric pressure at layer interfaces
    """
    top = xr.full_like(delp.isel({dim_center: [0]}), toa_pressure).variable
    delp_with_top = top.concat([top, delp.variable], dim=dim_center)
    pressure = delp_with_top.cumsum(dim_center)
    return _add_coords_to_interface_variable(
        pressure, delp, dim_center=dim_center
    ).rename({dim_center: dim_outer})


def height_at_interface(
    dz: xr.DataArray,
    phis: xr.DataArray,
    dim_center: str = COORD_Z_CENTER,
    dim_outer: str = COORD_Z_OUTER,
) -> xr.DataArray:
    """Compute geopotential height at layer interfaces.

    Args:
        dz: layer height thicknesses
        phis: surface geopotential
        dim_center (optional): name of vertical dimension for dz.
            Defaults to "pfull".
        dim_outer (optional): name of vertical dimension for output heights.
            Defaults to "phalf".

    Returns:
        height at layer interfaces
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
        return xr.DataArray(dv_outer, coords=da_center.drop_vars(dim_center).coords)
    else:
        return xr.DataArray(dv_outer, coords=da_center.coords)


def pressure_at_midpoint(
    delp: xr.DataArray, toa_pressure: float = _TOA_PRESSURE, dim: str = COORD_Z_CENTER
) -> xr.DataArray:
    """Compute pressure at layer midpoints by linear interpolation.

    Args:
        delp: pressure thicknesses
        toa_pressure (optional): pressure at the top of atmosphere.
            Defaults to 300Pa.
        dim (optional): name of vertical dimension for delp. Defaults to "pfull".

    Returns:
        atmospheric pressure at layer midpoints
    """
    pi = pressure_at_interface(delp, toa_pressure=toa_pressure, dim_center=dim)
    return _interface_to_midpoint(pi, dim_center=dim)


def height_at_midpoint(
    dz: xr.DataArray, phis: xr.DataArray, dim: str = COORD_Z_CENTER
) -> xr.DataArray:
    """Compute geopotential height at layer midpoints by linear interpolation.

    Args:
        dz: layer height thicknesses
        phis: surface geopotential
        dim (optional): name of vertical dimension for dz. Defaults to "pfull".

    Returns:
        height at layer midpoints
    """
    zi = height_at_interface(dz, phis, dim_center=dim)
    return _interface_to_midpoint(zi, dim_center=dim)


def _interface_to_midpoint(da, dim_center=COORD_Z_CENTER, dim_outer=COORD_Z_OUTER):
    da_mid = (
        da.isel({dim_outer: slice(0, -1)}) + da.isel({dim_outer: slice(1, None)})
    ) / 2
    return da_mid.rename({dim_outer: dim_center})


def pressure_at_midpoint_log(
    delp: xr.DataArray, toa_pressure: float = _TOA_PRESSURE, dim: str = COORD_Z_CENTER
) -> xr.DataArray:
    """Compute pressure at layer midpoints following Eq. 3.17 of Simmons
    and Burridge (1981), MWR.

    Args:
        delp: pressure thicknesses
        toa_pressure (optional): pressure at the top of atmosphere.
            Defaults to 300Pa.
        dim (optional): name of vertical dimension for delp. Defaults to "pfull".

    Returns:
        atmospheric pressure at layer midpoints
    """
    pi = pressure_at_interface(
        delp, toa_pressure=toa_pressure, dim_center=dim, dim_outer=dim
    )
    dlogp = np.log(pi).diff(dim)
    output = delp / dlogp
    # ensure that output chunks are the same
    # pressure at interface adds a single chunk at the model surface
    if delp.chunks:
        original_chunk_size = delp.chunks[delp.get_axis_num(dim)]
        output = output.chunk({dim: original_chunk_size})

    return output


def dz_and_top_to_phis(
    top_height: xr.DataArray, dz: xr.DataArray, dim: str = COORD_Z_CENTER
) -> xr.DataArray:
    """ Compute surface geopotential from model top height and layer thicknesses"""
    return _GRAVITY * (top_height + dz.sum(dim=dim))


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
