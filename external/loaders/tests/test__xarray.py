import doctest
import loaders.mappers._xarray
import xarray as xr
import numpy as np
import cftime

# ensure that imported path exists
from loaders.mappers import XarrayMapper, open_zarr  # noqa


def test_xarray_wrapper_doctests():
    doctest.testmod(loaders.mappers._xarray, raise_on_error=True)


def test_open_zarr(tmpdir):
    time = cftime.DatetimeJulian(2020, 1, 1)
    time_str = "20200101.000000"
    ds = xr.Dataset(
        {"a": (["time", "tile", "z", "y", "x"], np.ones((1, 2, 3, 4, 5)))},
        coords={"time": [time]},
    )
    ds.to_zarr(str(tmpdir), consolidated=True)

    mapper = open_zarr(str(tmpdir))
    assert isinstance(mapper, XarrayMapper)
    xr.testing.assert_equal(mapper[time_str], ds.isel(time=0).drop_vars(names="time"))
