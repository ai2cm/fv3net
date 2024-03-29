import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.delayed import delayed
import fsspec
from unittest.mock import Mock

import vcm
from vcm.xarray_loaders import to_json, open_json, dataset_from_dict


def assert_attributes_equal(a, b):
    for variable in a:
        assert a[variable].attrs == b[variable].attrs
    assert a.attrs == b.attrs


@pytest.fixture()
def ds():

    coords = {
        "tile": np.arange(6),
        "x": np.arange(48),
        "y": np.arange(48),
    }

    ds = xr.Dataset(
        {"a": (["tile", "y", "x"], np.random.sample((6, 48, 48)))},
        coords=coords,
        attrs={"foo": "bar"},
    )

    ds["a"].attrs["foo"] = "var"

    return ds


def test_open_tiles(tmpdir, ds):
    for i in range(6):
        ds.isel(tile=i).to_netcdf(tmpdir.join(f"prefix.tile{i+1}.nc"))

    loaded = vcm.open_tiles(str(tmpdir.join("prefix"))).load()

    xr.testing.assert_equal(ds, loaded)
    assert_attributes_equal(ds, loaded)


def test_open_tiles_errors_with_wrong_number_of_tiles(tmpdir, ds):
    for i in range(4):
        ds.isel(tile=i).to_netcdf(tmpdir.join(f"prefix.tile{i+1}.nc"))

    with pytest.raises(ValueError):
        vcm.open_tiles(str(tmpdir.join("prefix")))


@pytest.fixture()
def dataset():
    arr = np.random.rand(100, 10)
    coords = dict(time=np.arange(100), x=np.arange(10),)
    return xr.Dataset(
        {"a": (["time", "x"], arr), "b": (["time", "x"], arr)}, coords=coords
    )


def test_open_delayed(dataset):
    a_delayed = delayed(lambda x: x)(dataset)
    ds = vcm.open_delayed(a_delayed, schema=dataset)

    xr.testing.assert_equal(dataset, ds.compute())
    assert_attributes_equal(dataset, ds)
    assert isinstance(ds["a"].data, dask.array.Array)


def test_open_delayed_fills_nans(dataset):
    ds_no_b = dataset[["a"]]
    # wrap idenity with delated object
    a_delayed = delayed(lambda x: x)(ds_no_b)
    ds = vcm.open_delayed(a_delayed, schema=dataset)

    # test that b is filled with anans
    b = ds["b"].compute()
    assert np.all(np.isnan(b))
    assert b.dims == dataset["b"].dims
    assert b.dtype == dataset["b"].dtype


def test_dump_nc(tmpdir):
    ds = xr.Dataset({"a": (["x"], [1.0])})

    path = str(tmpdir.join("data.nc"))
    with fsspec.open(path, "wb") as f:
        vcm.dump_nc(ds, f)

    ds_compare = xr.open_dataset(path)
    xr.testing.assert_equal(ds, ds_compare)


def test_dump_nc_no_seek():
    """
    GCSFS file objects raise an error when seek is called in write mode::

        if not self.mode == "rb":
            raise ValueError("Seek only available in read mode")
            ValueError: Seek only available in read mode

    """
    ds = xr.Dataset({"a": (["x"], [1.0])})
    m = Mock()

    vcm.dump_nc(ds, m)
    m.seek.assert_not_called()


def multitype_dataset(include_times=None):
    np.random.seed(0)

    x = range(1, 6)
    if include_times:
        times = pd.date_range("2000", periods=7)
        floats_coords = {"time": times, "x": x}
    else:
        floats_coords = {"x": x}

    floats = xr.DataArray(
        np.random.random((5, 7)),
        dims=["x", "time"],
        coords=floats_coords,
        name="floats",
        attrs={"units": "mm/day"},
    )
    integers = xr.DataArray(range(1, 6), coords=[x], dims=["x"], name="integers")
    strings = xr.DataArray(list("abcde"), coords=[x], dims=["x"], name="strings")
    bools = xr.DataArray(np.ones(5, dtype=bool), coords=[x], dims=["x"], name="bools")
    return xr.merge([floats, integers, strings, bools]).assign_attrs(metadata="baz")


@pytest.mark.parametrize("include_times", [False, True])
def test_json_roundtrip(tmpdir, include_times):
    ds = multitype_dataset(include_times=include_times)
    filename = tmpdir.join("example.json")

    if include_times:
        with pytest.raises(NotImplementedError, match="not currently possible"):
            to_json(ds, filename)
    else:
        to_json(ds, filename)
        roundtrip = open_json(filename)
        xr.testing.assert_identical(roundtrip, ds)


@pytest.mark.parametrize("include_times", [False, True])
def test_from_dict(include_times):
    ds = multitype_dataset(include_times=include_times)
    roundtrip = dataset_from_dict(ds.to_dict())
    xr.testing.assert_identical(roundtrip, ds)


def test_to_json_passes_kwargs(tmpdir):
    ds = multitype_dataset(include_times=False)
    filename = tmpdir.join("example.json")
    to_json(ds, filename, indent=4)
    with open(filename, "r") as file:
        lines = file.readlines()
    assert len(lines) > 1
