import numpy as np
import xarray as xr
from vcm.convenience import open_delayed
from dask.delayed import delayed
from dask.array import Array


def test_open_delayed():
    arr = np.random.rand(100, 10)
    coords = dict(time=np.arange(100), x=np.arange(10),)
    a = xr.Dataset({"a": (["time", "x"], arr)}, coords=coords)

    a_delayed = delayed(lambda x: x)(a)
    ds = open_delayed(a_delayed, meta=a)

    for key in ds:
        assert isinstance(ds[key].data, Array)
    xr.testing.assert_equal(a, ds.compute())
