import numpy as np
import xarray as xr

# following are defined as in FV3GFS model (see FV3/fms/constants/constants.f90)
gravity = 9.80665  # m /s2
Rd = 287.05  # J / K / kg

# default for restart file
VERTICAL_DIM = "zaxis_1"

REVERSE = slice(None, None, -1)

TOA_PRESSURE = 300.0  # Pa


def absolute_pressure_interface(dp, toa_pressure=TOA_PRESSURE, dim=VERTICAL_DIM):
    dpv = dp.variable
    top = 0 * dpv.isel({dim: [0]}) + toa_pressure
    dp_with_top = top.concat([top, dpv], dim=dim)
    return dp_with_top.cumsum(dim)


def absolute_height_interface(dz, phis, dim=VERTICAL_DIM):
    dzv = -dz.variable  # dz in model is negative
    bottom = (0 * dzv.isel({dim: [0]}) + phis.variable / gravity).transpose(*dzv.dims)
    dzv_with_bottom = bottom.concat([dzv, bottom], dim=dim)
    return dzv_with_bottom.isel({dim: REVERSE}).cumsum(dim).isel({dim: REVERSE})


def interface_to_center(ds, dim=VERTICAL_DIM):
    return (ds.isel({dim: slice(0, -1)}) + ds.isel({dim: slice(1, None)})) / 2


def dp_to_p(dp, dim=VERTICAL_DIM):
    pi = absolute_pressure_interface(dp, dim=dim)
    pc = interface_to_center(pi, dim=dim)
    return xr.DataArray(pc, coords=dp.coords)


def dz_to_z(dz, phis, dim=VERTICAL_DIM):
    zi = absolute_height_interface(dz, phis, dim=dim)
    zc = interface_to_center(zi, dim=dim)
    return xr.DataArray(zc, coords=dz.coords)


def hydrostatic_dz_with_logp(T, q, dp):
    pi = absolute_pressure_interface(dp)
    tv = (1 + 0.61 * q) * T
    dlogp = xr.DataArray(np.log(pi)).diff(VERTICAL_DIM)
    return -dlogp * Rd * tv / gravity
