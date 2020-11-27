from datetime import datetime, timedelta
import numpy as np
import xarray as xr

from fv3post import fregrid


def test_fregrid():
    time_coord = [datetime(2016, 1, 1) + timedelta(hours=t) for t in range(5)]
    da = xr.DataArray(np.zeros((5, 6, 48, 48)), dims=["time", "tile", "x", "y"])
    ds = xr.Dataset({"foo": da, "bar": da}).assign_coords(time=time_coord)
    ds_latlon = fregrid(ds, "x", "y")
    assert "latitude" in ds_latlon.dims
