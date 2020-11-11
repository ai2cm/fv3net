import datetime

import cftime
import pytest
import xarray as xr
import numpy as np
import dask.array as da


import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from budget.data import shift
from budget.pipeline import run, OpenTimeChunks
import budget.config
from budget.budgets import _compute_second_moment, storage


def test_OpenTimeChunks():
    def _data():
        shape = t, tile, x = (12, 6, 10)
        chunks_big = (12, 1, 1)
        chunks_small = (1, 1, 1)

        dims = ["time", "tile", "x"]
        coords = {dim: np.arange(n) for n, dim in zip(shape, dims)}

        arr_big_chunk = da.ones(shape, chunks=chunks_big, dtype=np.float32)
        arr_small_chunk = da.ones(shape, chunks=chunks_small, dtype=np.float32)
        return xr.Dataset(
            {
                "vulcan_omega_coarse": (dims, arr_big_chunk),
                "restart_field": (dims, arr_small_chunk),
            },
            coords=coords,
        )

    def _assert_no_time_or_tile(ds):
        assert set(ds.dims) & {"time", "tile"} == set()

    with TestPipeline() as p:
        chunks = (
            p | beam.Create([None]) | beam.Map(lambda _: _data()) | OpenTimeChunks()
        )

        chunks | "Assert no time or tile" >> beam.Map(_assert_no_time_or_tile)


@pytest.mark.regression
def test_run(data_dirs, tmpdir):
    diag_path, restart_path, gfsphysics_url = data_dirs

    output_path = str(tmpdir.join("out"))
    run(restart_path, diag_path, gfsphysics_url, output_path)
    ds = xr.open_mfdataset(f"{output_path}/*.nc", combine="by_coords")

    for variable in budget.config.VARIABLES_TO_AVERAGE | {
        "exposed_area",
        "area",
    }:
        assert variable in ds
        assert "long_name" in ds[variable].attrs
        assert "units" in ds[variable].attrs

    assert "history" in ds.attrs


def test_shift():
    initial_time = cftime.DatetimeJulian(
        year=2016, month=8, day=5, hour=12, minute=7, second=30
    )
    dt = datetime.timedelta(minutes=15)
    times = [initial_time, initial_time + dt, initial_time + 2 * dt]

    arr = [0.0, 1.0, 2.0]
    ds = xr.Dataset({"a": (["time"], arr)}, coords={"time": times})

    steps = ["begin", "middle", "end"]
    expected_times = [initial_time + dt / 2, initial_time + 3 * dt / 2]
    expected = xr.Dataset(
        {"a": (["step", "time"], [[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]])},
        coords={"time": expected_times, "step": steps},
    )
    shifted = shift(ds, dt=dt / 2)

    xr.testing.assert_equal(shifted, expected)


@pytest.mark.parametrize(
    ["a", "b", "expected"],
    [
        (
            xr.DataArray(3.0, name="a", attrs={"units": "m", "long_name": "a_long"}),
            xr.DataArray(2.0, name="b", attrs={"units": "m", "long_name": "b_long"}),
            xr.DataArray(
                6.0,
                name="a_b",
                attrs={"units": "m m", "long_name": "Product of a_long and b_long"},
            ),
        ),
        (
            xr.DataArray(3.0, name="a", attrs={"long_name": "a_long"}),
            xr.DataArray(2.0, name="b", attrs={"units": "m", "long_name": "b_long"}),
            xr.DataArray(
                6.0,
                name="a_b",
                attrs={"units": "m", "long_name": "Product of a_long and b_long"},
            ),
        ),
        (
            xr.DataArray(3.0, name="a", attrs={"units": "m"}),
            xr.DataArray(2.0, name="b", attrs={"units": "m", "long_name": "b_long"}),
            xr.DataArray(
                6.0,
                name="a_b",
                attrs={"units": "m m", "long_name": "Product of a and b_long"},
            ),
        ),
    ],
)
def test__compute_second_moment(a, b, expected):
    result = _compute_second_moment(a, b)
    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        (
            xr.DataArray(
                [0.0, 1.0],
                coords=[("step", ["begin", "end"])],
                name="a",
                attrs={"units": "m", "long_name": "a_long"},
            ),
            xr.DataArray(
                0.5,
                name="a_storage",
                attrs={"units": "m/s", "long_name": "Storage of a_long"},
            ),
        ),
        (
            xr.DataArray(
                [0.0, 1.0],
                coords=[("step", ["begin", "end"])],
                name="a",
                attrs={"long_name": "a_long"},
            ),
            xr.DataArray(
                0.5,
                name="a_storage",
                attrs={"units": "/s", "long_name": "Storage of a_long"},
            ),
        ),
        (
            xr.DataArray([0.0, 1.0], coords=[("step", ["begin", "end"])], name="a"),
            xr.DataArray(
                0.5,
                name="a_storage",
                attrs={"units": "/s", "long_name": "Storage of a"},
            ),
        ),
    ],
)
def test_storage(field, expected):
    result = storage(field, 2)
    xr.testing.assert_identical(result, expected)
