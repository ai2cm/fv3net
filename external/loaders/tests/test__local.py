import xarray as xr

from loaders.mappers import LocalMapper


def test_LocalMapper(tmpdir):
    path = "a.nc"
    xr.Dataset().to_netcdf(str(tmpdir.join(path)))

    mapper = LocalMapper(str(tmpdir))

    assert set(mapper) == {"a"}


def test_LocalMapper_getitem(tmpdir):
    path = "a.nc"
    xr.Dataset().to_netcdf(str(tmpdir.join(path)))
    mapper = LocalMapper(str(tmpdir))
    assert isinstance(mapper["a"], xr.Dataset)
