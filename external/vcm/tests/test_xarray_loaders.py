import dask
import numpy as np
import pytest
import xarray as xr
from dask.delayed import delayed

import vcm


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

    loaded = vcm.open_tiles(str(tmpdir.join("prefix")))

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
