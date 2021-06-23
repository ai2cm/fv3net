from typing import Tuple

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
    coords = {coord_name: bins[:-1]}
    width = bins[1:] - bins[:-1]
    width_da = xr.DataArray(width, coords=coords, dims=[coord_name])
    count_da = xr.DataArray(count, coords=coords, dims=[coord_name])
    if "units" in da.attrs:
        count_da[coord_name].attrs["units"] = da.units
        width_da[coord_name].attrs["units"] = da.units
        width_da.attrs["units"] = da.units
    return count_da, width_da
