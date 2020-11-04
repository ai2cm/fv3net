import fsspec
import numpy as np
import xarray as xr

from consolidate_metadata import consolidate_metadata


def test_consolidate_metadata(tmpdir):
    fs = fsspec.filesystem("file")
    da = xr.DataArray(np.reshape(np.arange(20), (5, 4)), dims=("x", "t"))
    ds = xr.Dataset({"a": da.assign_attrs(units="m"), "b": da})
    path = str(tmpdir.join("ds.zarr"))
    ds.to_zarr(path)
    consolidate_metadata(fs, path)
    ds_consolidated = xr.open_zarr(path, consolidated=True)
    xr.testing.assert_identical(ds, ds_consolidated)
