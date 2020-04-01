import unittest
import fv3net
import xarray as xr
import zarr
import numpy as np
from itertools import product

import pytest


@pytest.mark.parametrize("dtype, fill_value", [(int, -1), (float, np.nan)])
def test_zarr_mapping_init_coord_fill_value(dtype, fill_value):
    arr = np.array([2.0], dtype=dtype)
    schema = xr.Dataset({"x": (["x"], arr)})

    coords = {"time": list("abc")}

    store = {}
    group = zarr.open_group(store)
    fv3net.ZarrMapping(group, schema, dims=["time"], coords=coords)

    # check that both are NaN since NaN != Nan
    if np.isnan(fill_value) and np.isnan(group["x"].fill_value):
        return
    assert group["x"].fill_value == fill_value


def test_zarr_mapping_set_1d(dtype=int):
    arr = np.array([2.0], dtype=dtype)
    schema = xr.Dataset({"a": (["x"], arr)}).chunk()
    coords = {"time": list("abc")}

    store = {}
    group = zarr.open_group(store)
    m = fv3net.ZarrMapping(group, schema, dims=["time"], coords=coords)
    m["a"] = schema
    m["b"] = schema
    m["c"] = schema

    ds = xr.open_zarr(store)
    for time in coords["time"]:
        a = ds.sel(time=time).drop("time").load()
        b = schema.load()
        xr.testing.assert_equal(a, b)


def test_zarr_mapping_set_2d(dtype=int):
    arr = np.array([2.0], dtype=dtype)
    schema = xr.Dataset({"a": (["x"], arr)}).chunk()
    coords = {"time": [0, 1, 2], "space": list("xyz")}

    store = {}
    group = zarr.open_group(store)
    m = fv3net.ZarrMapping(group, schema, dims=["time", "space"], coords=coords)
    for time, space in product(coords["time"], coords["space"]):
        m[time, space] = schema

    ds = xr.open_zarr(store)
    assert list(ds.a.dims) == ["time", "space", "x"]

    # check all data
    for time, space in product(coords["time"], coords["space"]):
        a = ds.sel(space=space, time=time).drop(["time", "space"]).load()
        b = schema.load()
        xr.testing.assert_equal(a, b)


if __name__ == "__main__":
    unittest.main()
