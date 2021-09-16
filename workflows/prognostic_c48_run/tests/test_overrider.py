import numpy as np
import xarray as xr

from runtime.overrider import DatasetCachedByChunk

da = xr.DataArray(
    np.arange(11 * 4 * 6).reshape((11, 4, 6)),
    dims=["time", "x", "y"],
    coords={"time": np.arange(2000, 2011)},
).chunk({"time": 5, "x": 4, "y": 4})
ds = xr.Dataset({"var": da})


def test_dataset_caching():
    cached = DatasetCachedByChunk(ds, "time")
    ds_year_2000 = cached.load(2000)
    xr.testing.assert_identical(ds_year_2000, ds.sel(time=2000))
    assert cached._load_chunk.cache_info().hits == 0
    cached.load(2001)
    assert cached._load_chunk.cache_info().hits == 1

