from typing import Any, Hashable, Mapping, Tuple

import numpy as np
import xarray as xr


def histogram(da: xr.DataArray, **kwargs) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute histogram and return tuple of counts and bin widths.
    
    Args:
        da: input data
        kwargs: optional parameters to pass on to np.histogram

    Return:
        counts, bin_widths tuple of xr.DataArrays. The coordinate of both arrays is
        equal to the left side of the histogram bins.
    """
    coord_name = f"{da.name}_bins" if da.name is not None else "bins"
    count, bins = np.histogram(da, **kwargs)
    coords: Mapping[Hashable, Any] = {coord_name: bins[:-1]}
    width = bins[1:] - bins[:-1]
    width_da = xr.DataArray(width, coords=coords, dims=[coord_name])
    count_da = xr.DataArray(count, coords=coords, dims=[coord_name])
    if "units" in da.attrs:
        count_da[coord_name].attrs["units"] = da.units
        width_da[coord_name].attrs["units"] = da.units
        width_da.attrs["units"] = da.units
    return count_da, width_da


def histogram2d(
    x: xr.DataArray, y: xr.DataArray, **kwargs
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Compute 2D histogram and return tuple of counts and bin widths.
    
    Args:
        x: input data
        y: input data
        kwargs: optional parameters to pass on to np.histogram

    Return:
        counts, x_bin_widths, y_bin_widths tuple of xr.DataArrays. The coordinate of all
        arrays is equal to the left side of the histogram bins.
    """
    xcoord_name = f"{x.name}_bins" if x.name is not None else "xbins"
    ycoord_name = f"{y.name}_bins" if y.name is not None else "ybins"
    count, xedges, yedges = np.histogram2d(x.values.ravel(), y.values.ravel(), **kwargs)
    xcoord: Mapping[Hashable, Any] = {xcoord_name: xedges[:-1]}
    ycoord: Mapping[Hashable, Any] = {ycoord_name: yedges[:-1]}
    xwidth = xedges[1:] - xedges[:-1]
    ywidth = yedges[1:] - yedges[:-1]
    xwidth_da = xr.DataArray(xwidth, coords=xcoord, dims=[xcoord_name])
    ywidth_da = xr.DataArray(ywidth, coords=ycoord, dims=[ycoord_name])
    count_da = xr.DataArray(
        count, coords={**xcoord, **ycoord}, dims=[xcoord_name, ycoord_name]
    )
    if "units" in x.attrs:
        xwidth_da[xcoord_name].attrs["units"] = x.units
        xwidth_da.attrs["units"] = x.units
    if "units" in y.attrs:
        ywidth_da[ycoord_name].attrs["units"] = y.units
        ywidth_da.attrs["units"] = y.units
    return count_da, xwidth_da, ywidth_da
