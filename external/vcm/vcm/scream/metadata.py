import xarray as xr
import logging
import numpy as np
from typing import Callable, Sequence

TIME_DIM_NAME = "time"

logger = logging.getLogger(__name__)


def _mask_invalid_value(ds: xr.Dataset) -> xr.Dataset:
    ds = xr.where(ds > 1e30, np.nan, ds)
    return ds


def _set_missing_attrs(ds: xr.Dataset) -> xr.Dataset:

    for var in ds:
        da = ds[var]

        # True for some prognostic zarrs
        if "description" in da.attrs and "long_name" not in da.attrs:
            da.attrs["long_name"] = da.attrs["description"]

        if "long_name" not in da.attrs:
            da.attrs["long_name"] = var

        if "units" not in da.attrs:
            da.attrs["units"] = "unspecified"
    return ds


def standardize_scream_diagnostics(
    ds: xr.Dataset, time: str = TIME_DIM_NAME
) -> xr.Dataset:
    """Standardize SCREAM diagnostics to a common format.

    Args:
        ds (xr.Dataset):
            Dataset of scream diagnostics outputs,
            presumably opened from disk via zarr.
        time (str, optional):
            Name of the time coordinate dimension in the dataset

    Returns:
        Data variable attributes ("long_name" and "units") are set.
        Clean up invalid values that hasn't been fixed upstream.
    """
    if time in ds.coords:
        ds[time].attrs["calendar"] = "julian"
    funcs: Sequence[Callable[[xr.Dataset], xr.Dataset]] = [
        xr.decode_cf,
        _set_missing_attrs,
        _mask_invalid_value,
    ]
    for func in funcs:
        ds = func(ds)
    return ds
