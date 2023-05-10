import numpy as np
import xarray as xr
from vcm.cubedsphere.constants import INIT_TIME_DIM, VAR_LON_CENTER
from typing import Sequence, Union


gravity = 9.81
specific_heat = 1004

HOUR_PER_DEG_LONGITUDE = 1.0 / 15


def timedelta_to_seconds(dt):
    one_second = np.timedelta64(1000000000, "ns")
    return dt / one_second


def local_time(ds, time=INIT_TIME_DIM, lon_var=VAR_LON_CENTER):
    fractional_hr = (
        ds[time].dt.hour + (ds[time].dt.minute / 60.0) + (ds[time].dt.second / 3600.0)
    )
    local_time = (fractional_hr + ds[lon_var] * HOUR_PER_DEG_LONGITUDE) % 24
    return local_time


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
    with xr.set_options(keep_attrs=True):
        return array.weighted(weights.fillna(0.0)).mean(dims)


def vertical_tapering_scale_factors(n_levels: int, cutoff: int, rate: float):
    z_arr = np.arange(n_levels)
    scaled = np.exp((z_arr[slice(None, cutoff)] - cutoff) / rate)
    unscaled = np.ones(n_levels - cutoff)
    return np.hstack([scaled, unscaled])
