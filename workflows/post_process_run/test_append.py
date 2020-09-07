import fsspec
import numpy as np
import xarray as xr
import append
import os
import zarr
from datetime import datetime
import cftime
import pytest


def _time_array(n, units):
    array = zarr.zeros((n))
    array[:] = np.arange(n)
    array.attrs["units"] = units
    array.attrs["calendar"] = "proleptic_gregorian"
    return array


@pytest.mark.parametrize(
    "source_attr, target_attr, expected_error",
    [
        ({}, {}, AttributeError),
        ({"calendar": "julian"}, {}, AttributeError),
        ({"calendar": "julian"}, {"calendar": "proleptic_gregorian"}, ValueError),
    ],
)
def test__assert_calendars_same(source_attr, target_attr, expected_error):
    source_array = zarr.zeros((5))
    for k, v in source_attr.items():
        source_array.attrs[k] = v
    target_array = zarr.zeros((5))
    for k, v in target_attr.items():
        target_array.attrs[k] = v
    with pytest.raises(expected_error):
        append._assert_calendars_same(source_array, target_array)


def test__set_array_time_units_like():
    source_array = _time_array(3, "days since 2016-08-08")
    target_array = _time_array(3, "days since 2016-08-05")
    append._set_array_time_units_like(source_array, target_array)
    assert source_array.attrs["units"] == target_array.attrs["units"]
    np.testing.assert_allclose(source_array[:], np.arange(3, 6))


def test__get_initial_timestamp(tmpdir):
    tmpdir.join("time_stamp.out").write("2016 8 1 3 0 0")
    timestamp = append._get_initial_timestamp(tmpdir)
    expected_timestamp = "20160801.030000"
    assert timestamp == expected_timestamp


@pytest.mark.parametrize(
    "shape, chunks, ax, shift, raises_value_error",
    [
        ((8,), (2,), 0, 4, False),
        ((8, 4), (2, 1), 0, 8, False),
        ((8, 4), (2, 2), 0, 8, False),
        ((8, 4), (2, 1), 0, 16, False),
        ((8, 4), (2, 1), 1, 1, False),
        ((8, 4), (2, 1), 1, 2, False),
        ((8, 4), (3, 1), 0, 8, True),
        ((8, 4), (2, 1), 0, 7, True),
    ],
)
def test__shift_array(tmpdir, shape, chunks, ax, shift, raises_value_error):
    path = str(tmpdir.join("test.zarr"))
    z1 = zarr.open(path, mode="w", shape=shape, chunks=chunks)
    z1[:] = np.zeros(shape)
    if raises_value_error:
        with pytest.raises(ValueError):
            append._shift_array(z1, ax, shift)
    else:
        items_before = os.listdir(path)
        append._shift_array(z1, ax, shift)
        items_after = os.listdir(path)
        assert len(items_before) == len(items_after)
        for item in items_before:
            if item != ".zarray":
                chunk_indices = item.split(".")
                chunk_indices[ax] = str(int(chunk_indices[ax]) + shift // chunks[ax])
                assert ".".join(chunk_indices) in items_after


@pytest.mark.parametrize(
    "with_coords, n_time, chunk_time, raises_value_error",
    [
        (True, 6, 2, False),
        (False, 6, 2, False),
        (True, 6, 1, False),
        (True, 6, 6, False),
        (True, 6, 4, True),
    ],
)
def test_append_zarr_along_time(
    tmpdir, with_coords, n_time, chunk_time, raises_value_error
):
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

    if raises_value_error:
        with pytest.raises(ValueError):
            append.append_zarr_along_time(path2, path1, fsspec.filesystem("file"))
    else:
        append.append_zarr_along_time(path2, path1, fsspec.filesystem("file"))
        manually_appended_ds = xr.open_zarr(path1, consolidated=True)
        expected_ds = xr.concat([ds1, ds2], dim="time")
        xr.testing.assert_identical(manually_appended_ds, expected_ds)


@pytest.mark.parametrize("datetime", [cftime.DatetimeJulian, datetime])
def test_append_zarr_along_time_cftime(tmpdir, datetime):
    ds = xr.Dataset(
        {"a": (["time"], np.arange(2))},
        coords={"time": [datetime(2000, 1, 1, 0), datetime(2000, 1, 1, 1)]},
    )

    path1 = str(tmpdir.join("ds1.zarr"))
    path2 = str(tmpdir.join("ds2.zarr"))

    ds.isel(time=slice(0, 1)).to_zarr(path1, consolidated=True)
    ds.isel(time=slice(1, 2)).to_zarr(path2, consolidated=True)

    append.append_zarr_along_time(path2, path1, fsspec.filesystem("file"), "time")

    raw_dataset = xr.open_zarr(path2, decode_cf=False)

    # this step will fail for metadata problems
    xr.decode_cf(raw_dataset)
