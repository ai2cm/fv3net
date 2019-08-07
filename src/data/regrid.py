import numpy as np
from scipy.interpolate import interp1d
from numba import jit
import xarray as xr
import xesmf as xe
from tqdm import tqdm


### Vertical interpolation
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
        output_core_dims=[[output_dim]]          
    )


def height_on_model_levels(data_3d):
    return interpolate_onto_coords_of_coords(
        data_3d.pres/100, data_3d.h_plev, input_dim='plev', output_dim='pfull')


### Horizontal interpolation
def regrid_horizontal(data_in, ddeg_out):
    """Interpolate horizontally from one rectangular grid to another
    Args:
      data_3d: Raw dataset to be regridded
      ddeg_out: Grid spacing of target grid in degrees
    """
    data_in = data_in.rename({'grid_xt': 'lon', 'grid_yt': 'lat'})

    # Create output dataset with appropriate lat-lon
    ds_out = xr.Dataset({
        'lon': (['lon'], np.arange(ddeg_out/2, 360, ddeg_out)),
        'lat': (['lat'], np.arange(-90+ddeg_out/2, 90, ddeg_out))
    })

    regridder = xe.Regridder(data_in, ds_out, 'bilinear', reuse_weights=True)
    
    # Regrid each variable in original dataset
    regridded_das = []
    for var in data_in:
        da = data_in[var]
        if 'lon' in da.coords and 'lat' in da.coords:
            regridded_das.append(regridder(da))
    return xr.Dataset({da.name: da for da in regridded_das})


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_3d')
    parser.add_argument('output_zarr')
    parser.add_argument('--ddeg_output', default=0.9375)

    args = parser.parse_args()

    data_3d = xr.open_zarr(args.data_3d)
    
    data_out = regrid_horizontal(data_3d, args.ddeg_output)

    data_out.to_zarr(args.output_zarr)

    
if __name__ == '__main__':
    main()

