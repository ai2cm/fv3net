import numpy as np
import xarray as xr
import zarr as zr
from numba import jit
from scipy.interpolate import interp1d

from vcm.convenience import open_dataset, replace_esmf_coords_reg_latlon

# import xesmf as xe



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
    coords, arg, output_dim="pfull", input_dim="plev"
):
    coord_1d = arg[input_dim]
    return xr.apply_ufunc(
        interpolate_1d_nd_target,
        coords,
        coord_1d,
        arg,
        input_core_dims=[[output_dim], [input_dim], [input_dim]],
        output_core_dims=[[output_dim]],
    )


def height_on_model_levels(data_3d):
    return interpolate_onto_coords_of_coords(
        data_3d.pres / 100, data_3d.h_plev, input_dim="plev", output_dim="pfull"
    )


def fregrid_bnds_to_esmf(grid_xt_bnds):
    """Convert GFDL fregrid bounds variables to ESMF compatible vector"""
    return np.hstack([grid_xt_bnds[:, 0], grid_xt_bnds[-1, 1]])


def fregrid_to_esmf_compatible_coords(data: xr.Dataset) -> xr.Dataset:
    """Add ESMF-compatible grid information
    
    GFDL's fregrid stores metadata about the coordinates in a different way than ESMF.
    This function adds lon, and lat coordinates as well as the bounding information 
    lon_b and lat_b.
    """
    data = data.rename({"grid_xt": "lon", "grid_yt": "lat"})

    lon_b = xr.DataArray(fregrid_bnds_to_esmf(data.grid_xt_bnds), dims=["lon_b"])
    lat_b = xr.DataArray(fregrid_bnds_to_esmf(data.grid_yt_bnds), dims=["lat_b"])

    return data.assign_coords(lon_b=lon_b, lat_b=lat_b)


### Horizontal interpolation
def regrid_horizontal(
    data_in,
    d_lon_out=1.0,
    d_lat_out=1.0,
    method="conservative",
    prev_regrid_dataset=None,
):
    """Interpolate horizontally from one rectangular grid to another
    
    Args:
      data_in: Raw dataset to be regridded
      d_lon_out: longitude grid spacing (in degrees)
      d_lat_out: latitude grid spacing (in degrees)
      method: ESMF regridding method to use
      prev_regrid_dataset: Dataset from previous regridding operation
        to compare with source data in.  If variables in source are
        not present in prev_regrid_dataset, then they will be regridded
        and added to the output.
    """
    raise NotImplementedError(
        "No longer using ESMF and this function is broken"
        " without a proper pip installer for esmpy."
    )

    data_in = fregrid_to_esmf_compatible_coords(data_in)

    contiguous_space = data_in.chunk({"lon": -1, "lat": -1, "time": 1})

    # Create output dataset with appropriate lat-lon
    # grid_out = xe.util.grid_global(d_lon_out, d_lat_out)

    # regridder = xe.Regridder(
    #     contiguous_space, grid_out, method, reuse_weights=True
    # )

    # Regrid each variable in original dataset
    regridded_das = []
    for var in contiguous_space:

        if prev_regrid_dataset is not None and var not in contiguous_space.coords:
            var_finished = _var_finished_regridding(prev_regrid_dataset, var)
            if var_finished:
                continue

        da = contiguous_space[var]
        if "lon" in da.coords and "lat" in da.coords:
            regridded_das.append(regridder(da))
    return xr.Dataset({da.name: da for da in regridded_das})


def _var_finished_regridding(prev_regrid_dataset: zr.hierarchy.Group, var: str):
    """ Checks if variable is present in output zarr and if it finished regridding."""
    if var in prev_regrid_dataset:
        prev_var = prev_regrid_dataset[var]
        is_finished_regrid = prev_var.nchunks == prev_var.nchunks_initialized

        if not is_finished_regrid:
            print(f"Removing partially regridded variable: {var}")
            prev_regrid_dataset.store.rmdir("/" + var)
    else:
        is_finished_regrid = False

    return is_finished_regrid


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_tag")
    parser.add_argument("output_zarr")
    parser.add_argument("-f", "--force_regrid", action="store_true")
    parser.add_argument("--data-3d", default=None)
    args = parser.parse_args()

    print("Opening data")
    if args.data_3d:
        data_3d = xr.open_zarr(args.data_3d)
    else:
        data_3d = open_dataset(args.dataset_tag)

    if args.force_regrid:
        print("Regridding all variables from target dataset.")
        partial_zarr_output = None
        open_mode = "w"
    else:
        open_mode = "a"
        try:
            partial_zarr_output = zr.open(args.output_zarr, mode="a")
            print(f"Regridding variables not already present in: {args.output_zarr}")
        except ValueError:
            # TODO: debug statement about not finding previous partial regrid output
            partial_zarr_output = None

    data_out = regrid_horizontal(data_3d, prev_regrid_dataset=partial_zarr_output)

    # TODO: Probably should put the coordinate change somewhere in regrid_horizontal
    data_out = replace_esmf_coords_reg_latlon(data_out)

    print("Output data:")
    print(data_out)
    data_out.to_zarr(args.output_zarr, mode=open_mode)


if __name__ == "__main__":
    main()
