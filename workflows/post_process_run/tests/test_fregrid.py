from datetime import datetime, timedelta
import numpy as np
import xarray as xr

from fv3post import fregrid


def test_fregrid():
    time = [datetime(2016, 1, 1) + timedelta(hours=t) for t in range(5)]
    da = xr.DataArray(np.ones((len(time), 6, 48, 48)), dims=["time", "tile", "x", "y"])
    ds = xr.Dataset({"foo": da, "bar": da}).assign_coords(time=time)
    ds_latlon = fregrid(ds, "x", "y")
    expected_sizes = {"time": len(time), "latitude": 180, "longitude": 360}
    assert dict(ds_latlon.sizes) == expected_sizes
