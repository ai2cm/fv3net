import numpy as np
import xarray as xr

# following are defined as in FV3GFS model (see FV3/fms/constants/constants.f90)
gravity = 9.80665  # m /s2
Rd = 287.05  # J / K / kg


VERTICAL_DIM = "zaxis_1"  # default for restart files
TOA_PRESSURE = 300.0  # Pa
REVERSE = slice(None, None, -1)


def pressure_on_interface(delp, toa_pressure=TOA_PRESSURE, dim=VERTICAL_DIM):
    delpv = delp.variable
    top = 0 * delpv.isel({dim: [0]}) + toa_pressure
    delp_with_top = top.concat([top, delpv], dim=dim)
    return delp_with_top.cumsum(dim)


def height_on_interface(dz, phis, dim=VERTICAL_DIM):
    dzv = -dz.variable  # dz in model is negative
    bottom = (0 * dzv.isel({dim: [0]}) + phis.variable / gravity).transpose(*dzv.dims)
    dzv_with_bottom = bottom.concat([dzv, bottom], dim=dim)
    return dzv_with_bottom.isel({dim: REVERSE}).cumsum(dim).isel({dim: REVERSE})


def interface_to_center(ds, dim=VERTICAL_DIM):
    return (ds.isel({dim: slice(0, -1)}) + ds.isel({dim: slice(1, None)})) / 2


def delp_to_p(delp, dim=VERTICAL_DIM):
    pi = pressure_on_interface(delp, dim=dim)
    pc = interface_to_center(pi, dim=dim)
    return xr.DataArray(pc, coords=delp.coords)


def dz_to_z(dz, phis, dim=VERTICAL_DIM):
    zi = height_on_interface(dz, phis, dim=dim)
    zc = interface_to_center(zi, dim=dim)
    return xr.DataArray(zc, coords=dz.coords)


def hydrostatic_dz(T, q, delp, dim=VERTICAL_DIM):
    pi = pressure_on_interface(delp, dim=dim)
    tv = (1 + 0.61 * q) * T
    dlogp = xr.DataArray(np.log(pi)).diff(dim)
    return -dlogp * Rd * tv / gravity


def dz_and_top_to_phis(top_height, dz, dim=VERTICAL_DIM):
    return gravity * (top_height + dz.sum(dim=dim))
