from datetime import datetime, timedelta
import numpy as np
import xarray as xr

from fv3post import FregridLatLon


def test_FregridLatLon():
    time = [datetime(2016, 1, 1) + timedelta(hours=t) for t in range(5)]
    da = xr.DataArray(np.ones((len(time), 6, 48, 48)), dims=["time", "tile", "y", "x"])
    ds = xr.Dataset({"foo": da, "bar": da}).assign_coords(time=time)
    fregridder = FregridLatLon("C48", 180, 360)
    ds_latlon = fregridder.regrid_scalar(ds, "x", "y")
    expected_sizes = {"bnds": 2, "time": len(time), "latitude": 180, "longitude": 360}
    assert dict(ds_latlon.sizes) == expected_sizes
