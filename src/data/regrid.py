import numpy as np
from scipy.interpolate import interp1d
from metpy.interpolate import interpolate_1d as metpy_interpolate
from numba import jit
import xarray as xr

from math import floor

def interpolate_1d_scipy(x, xp, arg):
    """simple test case"""
    return interp1d(xp, arg)(x)


@jit
def _interpolate_1d_2d(x, xp, arr):
    """
    Args:
      x: 2D
      xp: 1D
      arr: 2D

    Returns:
      output with same shape as x
    """

    assert x.shape[0] == arr.shape[0]
    n = x.shape[1]
    output = np.zeros_like(x)

    for k in range(arr.shape[0]):
        old_j = 0
        for i in range(n):
            # find lower boun
            for j in range(old_j, arr.shape[1] - 1):
                old_j = j
                if xp[j + 1] > x[k, i] >= xp[j]:
                    break
            # this will do linear extrapolation
            alpha = (x[k, i] - xp[j]) / (xp[j + 1] - xp[j])
            output[k, i] = arr[k, j + 1] * alpha + arr[k, j] * (1 - alpha)
    return output


def interpolate_1d_nd_target(x, xp, arr, axis=-1):
    """Interpolate a variable onto a new coordinate system

    Args:
      x: multi-dimensional array giving the coordinate xp as a function of the
        new coordinate.
      xp: coordinate along which the data is defined
      arr: data to interpolate, defined on grid given by xp.

    Keyword Args:
      axis: axis of arr along which xp is defined

    Returns:
      data interpolated onto the coordinates of x
    """
    x = np.swapaxes(x, axis, -1)
    arr = np.swapaxes(arr, axis, -1)

    xreshaped = x.reshape((-1, x.shape[-1]))
    arrreshaped = arr.reshape((-1, arr.shape[-1]))

    if axis < 0:
        axis = arr.ndim + axis
    matrix = _interpolate_1d_2d(xreshaped, xp, arrreshaped)
    reshaped = matrix.reshape(x.shape)
    return reshaped.swapaxes(axis, -1)


def interpolate_onto_coords_of_coords(
    coords, arg, output_dim='pfull', input_dim='plev'):
    coord_1d = arg[input_dim]
    return xr.apply_ufunc(
        interpolate_1d_nd_target,
        coords, coord_1d, arg,
        input_core_dims=[[output_dim], [input_dim], [input_dim]],
        output_core_dims=[[output_dim]],
        output_dtypes=[arg.dtype],
        dask='parallelized'
    )


def interpolate_1d(x, xp, *args, dim='pfull'):
    """Interpolate onto desired coordinate grid"""

    assert len(x.dims) == 1
    output_dim = x.dims[0]

    first_arg = args[0]

    return xr.apply_ufunc(
        metpy_interpolate, x, xp, *args,
        kwargs={'axis': -1},
        input_core_dims=[[output_dim]] + [[dim]] * (len(args) + 1),
        output_core_dims=[[output_dim]],
        output_dtypes=[arg.dtype for arg in args],
        dask='parallelized'
    )


def height_on_model_levels(data_3d):
    return interpolate_onto_coords_of_coords(
        data_3d.pres/100, data_3d.h_plev, input_dim='plev', output_dim='pfull')


@jit
def lagrangian_origin_coordinates(dx, dy, dz, u, v, w, h=1):
    """Semi-lagrangian advection term for data on (time, level, x, y) grid"""
    nt, nz, ny, nx = u.shape

    output = np.empty((u.ndim,) + u.shape)

    for t in range(nt):
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    alpha_x = - u[t, k, j, i] * h
                    alpha_y = - v[t, k, j, i] * h
                    alpha_z = - w[t, k, j, i] * h

                    dgrid_x = alpha_x / dx[j, i]
                    dgrid_y = alpha_y / dy[j, i]
                    dgrid_z = alpha_z / dz[t, k, j, i]

                    output[0, t, k, j, i] = t
                    output[1, t, k, j, i] = k + dgrid_z
                    output[2, t, k, j, i] = j + dgrid_y
                    output[3, t, k, j, i] = i + dgrid_x

    return output


@jit
def compute_dz(z):
    nt, nz, ny, nx = z.shape
    output = np.empty(z.shape)
    for t in range(nt):
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    if k == 0:
                        tmp = (z[t, 1, j, i] - z[t, 0, j, i])
                    elif k == nz - 1:
                        tmp = (z[t, -1, j, i] - z[t, -2, j, i])
                    else:
                        tmp = (z[t, k + 1, j, i] - z[t, k - 1, j, i]) / 2

                    output[t, k, j, i] = tmp

    return output


def compute_dx_dy(lat, lon):
    # convert lat/lon to meters
    radius_earth = 6378e3

    dlon = np.r_[lon[1] - lon[0], lon[2:] - lon[:-2], lon[-1] - lon[-2]]
    dlat = np.r_[lat[1] - lat[0], lat[2:] - lat[:-2], lat[-1] - lat[-2]]

    dx = np.cos(np.deg2rad(lat[:, np.newaxis])) * np.deg2rad(dlon) * radius_earth
    dy = np.deg2rad(dlat) * radius_earth
    dy = np.broadcast_to(dy[:, np.newaxis], (len(lat), len(lon)))

    return dx, dy


@jit
def lagrangian_update(dx, dy, dz, u, v, w, phi, h=1):
    """Semi-lagrangian advection term for data on (time, level, x, y) grid"""
    nt, nz, ny, nx = u.shape

    output = np.zeros_like(phi)

    for t in range(nt):
        for k in range(1, nz - 1):
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    alpha_x = - u[t, k, j, i] * h
                    alpha_y = - v[t, k, j, i] * h
                    alpha_z = - w[t, k, j, i] * h

                    dgrid_x = alpha_x / dx[j, i]
                    dgrid_y = alpha_y / dy[j, i]
                    dgrid_z = alpha_z / dz[t, k, j, i]

                    # new coordinates
                    k1 = k + dgrid_z
                    j1 = j + dgrid_y
                    i1 = i + dgrid_x

                    # bounding inds
                    k_left = floor(k1)
                    j_left = floor(j1)
                    i_left = floor(i1)

                    alpha_k = k1 - k_left
                    alpha_j = j1 - j_left
                    alpha_i = i1 - i_left

                    # linear interpolation
                    output[t, k, j, i] = (phi[t, k_left, j_left, i_left] * (1 - alpha_k - alpha_j - alpha_i)
                                          + phi[t, k_left + 1, j_left, i_left] * alpha_k
                                          + phi[t, k_left, j_left + 1, i_left] * alpha_j
                                          + phi[t, k_left, j_left, i_left + 1] * alpha_i)
    return output
