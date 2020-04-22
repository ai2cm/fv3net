import runfile
import zarr
import xarray as xr
import numpy as np
import fsspec
from multiprocessing.pool import ThreadPool

import pytest


def test_init_coord():

    time = np.array(
        [
            "2016-08-01T00:16:00.000000000",
            "2016-08-01T00:17:00.000000000",
            "2016-08-01T00:18:00.000000000",
            "2016-08-01T00:19:00.000000000",
            "2016-08-01T00:20:00.000000000",
            "2016-08-01T00:21:00.000000000",
            "2016-08-01T00:22:00.000000000",
            "2016-08-01T00:23:00.000000000",
            "2016-08-01T00:24:00.000000000",
            "2016-08-01T00:25:00.000000000",
            "2016-08-01T00:26:00.000000000",
            "2016-08-01T00:27:00.000000000",
            "2016-08-01T00:28:00.000000000",
            "2016-08-01T00:29:00.000000000",
            "2016-08-01T00:30:00.000000000",
        ],
        dtype="datetime64[ns]",
    )

    ds = xr.Dataset(coords={"time": time})
    time = runfile._get_forecast_time(ds.time)

    ds_lead_time = ds.assign(time=time)

    store = {}

    group = zarr.open_group(store, mode="w")
    runfile.init_coord(group, ds_lead_time["time"])

    loaded = xr.open_zarr(store)
    np.testing.assert_equal(loaded.time.values, ds_lead_time.time.values)


@pytest.fixture(params=["gcsfs", "memory"])
def dest(request):
    if request.param == "gcsfs":
        return fsspec.get_mapper("gs://vcm-ml-data/testing-noah/temporarydeleteme/")
    elif request.param == "memory":
        return zarr.MemoryStore()


def test__cached_store_zarr_integration():
    src = zarr.MemoryStore()
    dest = zarr.MemoryStore()
    with ThreadPool(10) as pool:
        cache = runfile.CachedStore(src, dest, pool)
        zarr.open_group(cache, mode="w")


def test__cached_store_insert():
    src = {}
    dest = {}
    with ThreadPool(10) as pool:
        cache = runfile.CachedStore(src, dest, pool, cache_size=2)
        cache[0] = 0
        cache[1] = 1
        cache[2] = 2
    assert dict(src) == {}
    assert len(dest) == 3
