import datetime

import cftime
import pytest
import xarray as xr

from budget.data import shift
from budget.pipeline import run
import budget.config
from budget.budgets import _compute_second_moment, storage


@pytest.mark.regression
def test_run(data_dirs, tmpdir):
    diag_path, restart_path, gfsphysics_url, area_url = data_dirs

    output_path = str(tmpdir.join("out"))
    run(restart_path, diag_path, gfsphysics_url, area_url, output_path)
    ds = xr.open_zarr(output_path)

    for variable in budget.config.VARIABLES_TO_AVERAGE | {
        "exposed_area",
        "area",
    }:
        assert variable in ds
        assert "long_name" in ds[variable].attrs
        assert "units" in ds[variable].attrs

    # TODO add history back in
    # assert "history" in ds.attrs


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
