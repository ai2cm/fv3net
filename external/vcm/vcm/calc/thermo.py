import numpy as np
import xarray as xr
from ..cubedsphere.constants import COORD_Z_CENTER

# following are defined as in FV3GFS model (see FV3/fms/constants/constants.f90)
GRAVITY = 9.80665  # m /s2
RDGAS = 287.05  # J / K / kg
RVGAS = 461.5  # J / K / kg


TOA_PRESSURE = 300.0  # Pa
REVERSE = slice(None, None, -1)


def pressure_at_interface(delp, toa_pressure=TOA_PRESSURE, dim=COORD_Z_CENTER):
    """ Compute pressure at layer interfaces """
    delpv = delp.variable
    top = 0 * delpv.isel({dim: [0]}) + toa_pressure
    delp_with_top = top.concat([top, delpv], dim=dim)
    return delp_with_top.cumsum(dim)


def height_at_interface(dz, phis, dim=COORD_Z_CENTER):
    """ Compute geopotential height at layer interfaces """
    dzv = -dz.variable  # dz in model is negative
    bottom = (0 * dzv.isel({dim: [0]}) + phis.variable / GRAVITY).transpose(*dzv.dims)
    dzv_with_bottom = bottom.concat([dzv, bottom], dim=dim)
    return dzv_with_bottom.isel({dim: REVERSE}).cumsum(dim).isel({dim: REVERSE})


def pressure_at_midpoint(delp, dim=COORD_Z_CENTER):
    """ Compute pressure at layer midpoints """
    pi = pressure_at_interface(delp, dim=dim)
    pc = _interface_to_midpoint(pi, dim=dim)
    return xr.DataArray(pc, coords=delp.coords)


def height_at_midpoint(dz, phis, dim=COORD_Z_CENTER):
    """ Compute geopotential height at layer midpoints """
    zi = height_at_interface(dz, phis, dim=dim)
    zc = _interface_to_midpoint(zi, dim=dim)
    return xr.DataArray(zc, coords=dz.coords)


def _interface_to_midpoint(ds, dim=COORD_Z_CENTER):
    return (ds.isel({dim: slice(0, -1)}) + ds.isel({dim: slice(1, None)})) / 2


def pressure_at_midpoint_log(delp, dim=COORD_Z_CENTER):
    """ Compute pressure at layer midpoints following Eq. 3.17 of Simmons
    and Burridge (1981), MWR."""
    pi = pressure_at_interface(delp, dim=dim)
    dlogp = xr.DataArray(np.log(pi)).diff(dim)
    return xr.DataArray(delp / dlogp, coords=delp.coords)


def hydrostatic_dz(T, q, delp, dim=COORD_Z_CENTER):
    """ Compute layer thickness assuming hydrostatic balance """
    pi = pressure_at_interface(delp, dim=dim)
    epsilon = RDGAS / RVGAS
    tv = T * (1 + q / epsilon) / (1 + q)
    #tv = (1 + 0.61 * q) * T
    dlogp = xr.DataArray(np.log(pi)).diff(dim)
    return -dlogp * RDGAS * tv / GRAVITY


def dz_and_top_to_phis(top_height, dz, dim=COORD_Z_CENTER):
    """ Compute surface geopotential from model top height and layer thicknesses """
    return GRAVITY * (top_height + dz.sum(dim=dim))
