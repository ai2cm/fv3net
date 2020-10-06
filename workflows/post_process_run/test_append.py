import fsspec
import numpy as np
import xarray as xr
import append
import os
import zarr
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
    "with_coords, lengths, chunk_sizes, raises_value_error",
    [
        (False, (6, 6), (2, 2), False),
        (True, (6, 6), (2, 2), False),
        (False, (6, 4), (2, 2), False),
        (True, (6, 4), (2, 2), False),
        (False, (6, 5), (2, 2), False),
        (True, (6, 5), (2, 2), False),
        (True, (5, 6), (2, 2), True),
        (True, (6, 6), (3, 2), True),
        (True, (6, 6), (4, 2), True),
    ],
)
def test_append_zarr_along_time(
    tmpdir, with_coords, lengths, chunk_sizes, raises_value_error,
):
    fs = fsspec.filesystem("file")
    # generate test datasets
    datasets = []
    for length, chunk_size in zip(lengths, chunk_sizes):
        array = xr.DataArray(
            np.arange(5 * length).reshape((length, 5)), dims=["time", "x"]
        )
        ds = xr.Dataset({"var1": array.chunk({"time": chunk_size})})
        ds["var1"].encoding["chunks"] = (chunk_size, 5)
        datasets.append(ds)

    if with_coords:
        full_coord = [
            cftime.DatetimeJulian(2000, 1, d) for d in range(1, sum(lengths) + 1)
        ]
        for i, ds in enumerate(datasets):
            ds_coord = full_coord[sum(lengths[:i]) : sum(lengths[: i + 1])]
            datasets[i] = ds.assign_coords(time=ds_coord)
            datasets[i]["time"].encoding["chunks"] = (chunk_sizes[i],)

    paths = [str(tmpdir.join(f"ds{i}.zarr")) for i in range(len(datasets))]
    for ds, path in zip(datasets, paths):
        ds.to_zarr(path, consolidated=True)

    # append zarrs using append_zarr_along_time
    if raises_value_error:
        with pytest.raises(ValueError):
            append.append_zarr_along_time(paths[1], paths[0], fs)
    else:
        append.append_zarr_along_time(paths[1], paths[0], fs)
        expected_ds = xr.concat(datasets, dim="time")
        manually_appended_ds = xr.open_zarr(paths[0], consolidated=True)
        xr.testing.assert_identical(manually_appended_ds, expected_ds)
