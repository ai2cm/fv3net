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
    n_time = 6
    chunk_time = 2
    da = xr.DataArray(np.arange(5 * n_time).reshape((n_time, 5)), dims=["time", "x"])
    ds = xr.Dataset({"var1": da.chunk({"time": chunk_time})})
    if with_coords:
        coord1 = [datetime(2000, 1, d) for d in range(1, 1 + n_time)]
        coord2 = [datetime(2000, 1, d) for d in range(1 + n_time, 1 + 2 * n_time)]
        ds1 = ds.assign_coords(time=coord1)
        ds2 = ds.assign_coords(time=coord2)
    else:
        ds1 = ds.copy()
        ds2 = ds.copy()

    path1 = str(tmpdir.join("ds1.zarr"))
    path2 = str(tmpdir.join("ds2.zarr"))

    ds1.to_zarr(path1, consolidated=True)
    ds2.to_zarr(path2, consolidated=True)

    if with_coords:
        append_run.encode_time_units_like(path2, path1)
    append_run.shift_store(path2, "time", n_time)

    copytree(path2, path1)
    zarr.consolidate_metadata(path1)

    manually_appended_ds = xr.open_zarr(path1, consolidated=True)
    expected_ds = xr.concat([ds1, ds2], dim="time")

    xr.testing.assert_allclose(manually_appended_ds, expected_ds)


def test__get_initial_timestamp(tmpdir):
    tmpdir.join("time_stamp.out").write("2016 8 1 3 0 0")
    timestamp = append_run._get_initial_timestamp(tmpdir)
    expected_timestamp = "20160801.030000"
    assert timestamp == expected_timestamp
