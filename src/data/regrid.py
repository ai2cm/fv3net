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


def fregrid_bnds_to_esmf(grid_xt_bnds):
    """Convert GFDL fregrid bounds variables to ESMF compatible vector"""
    return np.hstack([grid_xt_bnds[:,0], grid_xt_bnds[-1,1]])
    
    
def fregrid_to_esmf_compatible_coords(data: xr.Dataset) -> xr.Dataset:
    """Add ESMF-compatible grid information
    
    GFDL's fregrid stores metadata about the coordinates in a different way than ESMF.
    This function adds lon, and lat coordinates as well as the bounding information 
    lon_b and lat_b.
    """
    data = data.rename({'grid_xt': 'lon', 'grid_yt': 'lat'})
    
    lon_b = xr.DataArray(fregrid_bnds_to_esmf(data.grid_xt_bnds), dims=['lon_b'])
    lat_b = xr.DataArray(fregrid_bnds_to_esmf(data.grid_yt_bnds), dims=['lat_b'])
    
    return data.assign_coords(lon_b=lon_b, lat_b=lat_b)
    

### Horizontal interpolation
def regrid_horizontal(data_in, d_lon_out=1.0, d_lat_out=1.0, method='conservative'):
    """Interpolate horizontally from one rectangular grid to another
    
    Args:
      data_3d: Raw dataset to be regridded
      ddeg_out: Grid spacing of target grid in degrees
    """
    
    data_in = fregrid_to_esmf_compatible_coords(data_in)
    
    contiguous_space = data_in.chunk({'lon': -1, 'lat': -1, 'time': 1})
    
    # Create output dataset with appropriate lat-lon
    grid_out = xe.util.grid_global(d_lon_out, d_lat_out)
    
    regridder = xe.Regridder(contiguous_space, grid_out, method, reuse_weights=True)
    
    # Regrid each variable in original dataset
    regridded_das = []
    for var in contiguous_space:
        da = contiguous_space[var]
        if 'lon' in da.coords and 'lat' in da.coords:
            regridded_das.append(regridder(da))
    return xr.Dataset({da.name: da for da in regridded_das})


def main():
    from src.data import open_dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-3d', default=None)
    parser.add_argument('output_zarr')
    args = parser.parse_args()

    print("Opening data")
    if args.data_3d:
        data_3d = xr.open_zarr(args.data_3d)
    else:
        data_3d = open_dataset('gfdl')
        
    data_out = regrid_horizontal(data_3d)
    print("Output data:")
    print(data_out)
    data_out.to_zarr(args.output_zarr, mode='w')

    
if __name__ == '__main__':
    main()

