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
    """ Compute pressure at layer interfaces """
    top = xr.full_like(delp.isel({dim_center: [0]}), toa_pressure).variable
    delp_with_top = top.concat([top, delp.variable], dim=dim_center)
    return xr.DataArray(delp_with_top.cumsum(dim_center)).rename(
        {dim_center: dim_outer}
    )


def height_at_interface(dz, phis, dim_center=COORD_Z_CENTER, dim_outer=COORD_Z_OUTER):
    """ Compute geopotential height at layer interfaces """
    bottom = phis.broadcast_like(dz.isel({dim_center: [0]})) / GRAVITY
    dzv = -dz.variable  # dz is negative in model
    dz_with_bottom = dzv.concat([dzv, bottom], dim=dim_center)
    return xr.DataArray(
        dz_with_bottom.isel({dim_center: REVERSE})
        .cumsum(dim_center)
        .isel({dim_center: REVERSE})
    ).rename({dim_center: dim_outer})


def pressure_at_midpoint(delp, dim=COORD_Z_CENTER):
    """ Compute pressure at layer midpoints """
    pi = pressure_at_interface(delp, dim_center=dim)
    pc = _interface_to_midpoint(pi, dim_center=dim)
    return xr.DataArray(pc, coords=delp.coords)


def height_at_midpoint(dz, phis, dim=COORD_Z_CENTER):
    """ Compute geopotential height at layer midpoints """
    zi = height_at_interface(dz, phis, dim_center=dim)
    zc = _interface_to_midpoint(zi, dim_center=dim)
    return xr.DataArray(zc, coords=dz.coords)


def _interface_to_midpoint(da, dim_center=COORD_Z_CENTER, dim_outer=COORD_Z_OUTER):
    da_mid = (
        da.isel({dim_outer: slice(0, -1)}) + da.isel({dim_outer: slice(1, None)})
    ) / 2
    return da_mid.rename({dim_outer: dim_center})


def pressure_at_midpoint_log(delp, dim=COORD_Z_CENTER):
    """ Compute pressure at layer midpoints following Eq. 3.17 of Simmons
    and Burridge (1981), MWR."""
    pi = pressure_at_interface(delp, dim_center=dim)
    dlogp = xr.DataArray(np.log(pi)).diff(dim)
    return xr.DataArray(delp / dlogp, coords=delp.coords)


def hydrostatic_dz(T, q, delp, dim=COORD_Z_CENTER):
    """ Compute layer thickness assuming hydrostatic balance """
    pi = pressure_at_interface(delp, dim_center=dim)
    epsilon = RDGAS / RVGAS
    tv = T * (1 + q / epsilon) / (1 + q)
    # tv = (1 + 0.61 * q) * T
    dlogp = xr.DataArray(np.log(pi)).diff(dim)
    return -dlogp * RDGAS * tv / GRAVITY


def dz_and_top_to_phis(top_height, dz, dim=COORD_Z_CENTER):
    """ Compute surface geopotential from model top height and layer thicknesses """
    return GRAVITY * (top_height + dz.sum(dim=dim))
