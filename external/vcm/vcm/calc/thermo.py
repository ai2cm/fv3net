import numpy as np
import xarray as xr
from ..cubedsphere.constants import COORD_Z_CENTER, COORD_Z_OUTER

# following are defined as in FV3GFS model (see FV3/fms/constants/constants.f90)
GRAVITY = 9.80665  # m /s2
RDGAS = 287.05  # J / K / kg
RVGAS = 461.5  # J / K / kg


TOA_PRESSURE = 300.0  # Pa
REVERSE = slice(None, None, -1)


def pressure_at_interface(
    delp, toa_pressure=TOA_PRESSURE, dim_center=COORD_Z_CENTER, dim_outer=COORD_Z_OUTER
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
    return _add_coords(pressure, delp, dim_center=dim_center, dim_outer=dim_outer)


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
    bottom = phis.broadcast_like(dz.isel({dim_center: [0]})) / GRAVITY
    dzv = -dz.variable  # dz is negative in model
    dz_with_bottom = dzv.concat([dzv, bottom], dim=dim_center)
    height = (
        dz_with_bottom.isel({dim_center: REVERSE})
        .cumsum(dim_center)
        .isel({dim_center: REVERSE})
    )
    return _add_coords(height, dz, dim_center=dim_center, dim_outer=dim_outer)


def _add_coords(dv, da, dim_center=COORD_Z_CENTER, dim_outer=COORD_Z_OUTER):
    """ Assign coords from da to dv, except use vertical coordinate that is
    one element longer than da's, if vertical coordinate exists in da """
    if dim_center in da.coords:
        z_coord_center = da.coords[dim_center]
        z_coord_outer = xr.concat(
            [z_coord_center, 1 + z_coord_center.isel({dim_center: [-1]})],
            dim=dim_center,
        ).rename({dim_center: dim_outer})
        return (
            xr.DataArray(dv, coords=da.drop(dim_center).coords)
            .rename({dim_center: dim_outer})
            .assign_coords({dim_outer: z_coord_outer})
        )
    else:
        return xr.DataArray(dv, coords=da.coords).rename({dim_center: dim_outer})


def pressure_at_midpoint(delp, toa_pressure=TOA_PRESSURE, dim=COORD_Z_CENTER):
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
        da.isel({dim_outer: slice(0, -1)})
        + da.isel({dim_outer: slice(1, None)}).variable
    ) / 2
    return da_mid.rename({dim_outer: dim_center})


def pressure_at_midpoint_log(delp, toa_pressure=TOA_PRESSURE, dim=COORD_Z_CENTER):
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
    dlogp = np.log(pi).diff(dim, label="lower")
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
    tv = T * (1 + (RVGAS / RDGAS - 1) * q)
    dlogp = np.log(pi).diff(dim, label="lower")
    return -dlogp * RDGAS * tv / GRAVITY


def dz_and_top_to_phis(top_height, dz, dim=COORD_Z_CENTER):
    """ Compute surface geopotential from model top height and layer thicknesses"""
    return GRAVITY * (top_height + dz.sum(dim=dim))
