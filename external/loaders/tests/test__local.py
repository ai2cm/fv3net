import numpy as np
import xarray as xr

from loaders.mappers import LocalMapper, mapper_to_local


def test_LocalMapper(tmpdir):
    path = "a.nc"
    xr.Dataset({"a": (["x"], [1.0])}).to_netcdf(str(tmpdir.join(path)))

    mapper = LocalMapper(str(tmpdir))

    assert set(mapper) == {"a"}


def test_LocalMapper_getitem(tmpdir):
    path = "a.nc"
    xr.Dataset({"a": (["x"], [1.0])}).to_netcdf(str(tmpdir.join(path)))
    mapper = LocalMapper(str(tmpdir))
    ds = mapper["a"]

    assert isinstance(ds, xr.Dataset)
    assert isinstance(ds.a.data, np.ndarray)


def test_mapper_to_local(tmpdir):
    mapper = {"a": xr.Dataset({"a": (["x"], [1.0])})}

    local_mapper = mapper_to_local(mapper, str(tmpdir))
    xr.testing.assert_equal(mapper["a"], local_mapper["a"])
