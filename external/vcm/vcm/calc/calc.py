import numpy as np
import xarray as xr
from vcm.cubedsphere.constants import INIT_TIME_DIM, VAR_LON_CENTER
from typing import Sequence, Union


gravity = 9.81
specific_heat = 1004

HOUR_PER_DEG_LONGITUDE = 1.0 / 15


def apparent_heating(dtemp_dt, w):
    return dtemp_dt + w * gravity / specific_heat


def timedelta_to_seconds(dt):
    one_second = np.timedelta64(1000000000, "ns")
    return dt / one_second


def apparent_source(
    q: xr.DataArray, t_dim: str, s_dim: str, coarse_tstep_idx=0, highres_tstep_idx=0
) -> xr.DataArray:
    """Compute the apparent source from stepped output

    Args:
        q: The variable to compute the source of
        t_dim, optional: the dimension corresponding to the initial condition
        s_dim, optional: the dimension corresponding to the forecast time
        step_dim: dimension corresponding to the step time dimension
            (begin, before physics, after physics)
        coarse_tstep_idx: (default=0) forecast time step to use for
            calculating one step run tendency
        highres_tstep_idx: (default=0) forecast time step to use for
            calculating high res run tendency
    Returns:
        The apparent source of q. Has units [q]/s

    """

    t = q[t_dim]
    s = q[s_dim]
    q = q.drop_vars([t_dim, s_dim])
    dq = q.diff(t_dim)
    dq_c48 = q.diff(s_dim)
    dt = timedelta_to_seconds(t.diff(t_dim))
    ds = timedelta_to_seconds(s.diff(s_dim))

    tend = dq / dt
    tend_c48 = dq_c48 / ds

    # restore coords
    tend = tend.isel({s_dim: highres_tstep_idx}).assign_coords(**{t_dim: t[:-1]})
    tend_c48 = tend_c48.isel(
        {s_dim: coarse_tstep_idx, t_dim: slice(0, -1)}
    ).assign_coords(**{t_dim: t[:-1]})

    return tend - tend_c48


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
    return xr.apply_ufunc(
        _weighted_average,
        array,
        weights,
        input_core_dims=[dims, dims],
        kwargs=kwargs,
        dask="allowed",
    )
