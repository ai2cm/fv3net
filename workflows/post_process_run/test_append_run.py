import numpy as np
import xarray as xr
import append_run
import os
import shutil
import zarr
from datetime import datetime
import pytest


def copytree(src, dst):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d)
        else:
            shutil.copy(s, d)


@pytest.mark.parametrize("with_coords", [True, False])
def test_appending_shifted_zarr_gives_expected_ds(tmpdir, with_coords):
    da = xr.DataArray(np.arange(20).reshape((4, 5)), dims=["time", "x"])
    ds = xr.Dataset({"var1": da.chunk({"time": 2})})
    if with_coords:
        coord1 = [datetime(2000, 1, d) for d in range(1, 5)]
        coord2 = [datetime(2000, 1, d) for d in range(5, 9)]
        ds1 = ds.assign_coords(time=coord1)
        ds2 = ds.assign_coords(time=coord2)
    else:
        ds1 = ds.copy()
        ds2 = ds.copy()

    path1 = str(tmpdir.join("ds1.zarr"))
    path2 = str(tmpdir.join("ds2.zarr"))

    ds1.to_zarr(path1, consolidated=True)
    ds2.to_zarr(path2, consolidated=True)

    append_run._shift_chunks(path2, "time", 2)

    copytree(path2, path1)
    zarr.consolidate_metadata(path1)

    appended_ds = xr.open_zarr(path1, consolidated=True)

    assert appended_ds.sizes["time"] == 2 * ds.sizes["time"]
    if with_coords:
        assert (appended_ds.time.values == coord1 + coord2).all()
