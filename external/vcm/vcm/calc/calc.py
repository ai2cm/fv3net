import numpy as np
import xarray as xr
from vcm.cubedsphere.constants import INIT_TIME_DIM, VAR_LON_CENTER
from typing import Sequence, Union, Optional


gravity = 9.81
specific_heat = 1004

HOUR_PER_DEG_LONGITUDE = 1.0 / 15

XRData = Union[xr.DataArray, xr.Dataset]


def average_over_dims(x: XRData, dims=["x", "y", "tile"]) -> XRData:
    return x.mean(dims)


def zonal_average(
    x: XRData,
    lat: xr.DataArray,
    bins: Optional[Sequence[float]] = None,
    lat_name: str = "lat",
) -> XRData:
    bins = bins or np.arange(-90, 91, 2)
    with xr.set_options(keep_attrs=True):
        output = x.groupby_bins(lat.rename("lat"), bins=bins).mean()
        output = output.rename({"lat_bins": lat_name})
    lats_mid = [lat.item().mid for lat in output[lat_name]]
    return output.assign_coords({lat_name: lats_mid})


def timedelta_to_seconds(dt):
    one_second = np.timedelta64(1000000000, "ns")
    return dt / one_second


def local_time(ds, time=INIT_TIME_DIM, lon_var=VAR_LON_CENTER):
    fractional_hr = (
        ds[time].dt.hour + (ds[time].dt.minute / 60.0) + (ds[time].dt.second / 3600.0)
    )
    local_time = (fractional_hr + ds[lon_var] * HOUR_PER_DEG_LONGITUDE) % 24
    return local_time


def _weighted_average(array, weights, axis=None):

    return np.nansum(array * weights, axis=axis) / np.nansum(weights, axis=axis)


def weighted_average(
    array: Union[xr.Dataset, xr.DataArray],
    weights: xr.DataArray,
    dims: Sequence[str] = ["tile", "y", "x"],
) -> xr.Dataset:
    """Compute a weighted average of an array or dataset

    Args:
        array: xr dataarray or dataset of variables to averaged
        weights: xr datarray of grid cell weights for averaging
        dims: dimensions to average over

    Returns:
        xr dataarray or dataset of weighted averaged variables
    """
    if dims is not None:
        kwargs = {"axis": tuple(range(-len(dims), 0))}
    else:
        kwargs = {}
    with xr.set_options(keep_attrs=True):
        return xr.apply_ufunc(
            _weighted_average,
            array,
            weights,
            input_core_dims=[dims, dims],
            kwargs=kwargs,
            dask="allowed",
        )


def vertical_tapering_scale_factors(n_levels: int, cutoff: int, rate: float):
    z_arr = np.arange(n_levels)
    scaled = np.exp((z_arr[slice(None, cutoff)] - cutoff) / rate)
    unscaled = np.ones(n_levels - cutoff)
    return np.hstack([scaled, unscaled])
