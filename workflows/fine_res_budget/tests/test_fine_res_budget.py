import datetime
from pathlib import Path

import cftime
import pytest
import xarray as xr
import numpy as np
import dask.array as da

import synth

import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from budget.data import shift
from budget.pipeline import run, OpenTimeChunks
from budget.budgets import _compute_second_moment, storage

from vcm import safe

SCHEMAS = {
    "restart": "restart.json",
    "atmos_15min_coarse_ave": "atmos_15min_coarse_ave_schema.json",
    "gfsphysics_15min_coarse": "gfsphysics_15min_coarse_schema.json",
}
VARIABLES = {
    "restart": [
        "grid_x",
        "grid_y",
        "grid_xt",
        "grid_yt",
        "pfull",
        "tile",
        "delp",
        "T",
        "sphum",
    ],
    "atmos_15min_coarse_ave": [
        "t_dt_fv_sat_adj_coarse",
        "t_dt_nudge_coarse",
        "t_dt_phys_coarse",
        "qv_dt_fv_sat_adj_coarse",
        "qv_dt_phys_coarse",
        "eddy_flux_vulcan_omega_sphum",
        "eddy_flux_vulcan_omega_temp",
        "grid_latt_coarse",
        "grid_lont_coarse",
        "grid_lat_coarse",
        "grid_lon_coarse",
        "vulcan_omega_coarse",
        "area_coarse",
    ],
    "gfsphysics_15min_coarse": [
        "grid_latt_coarse",
        "grid_lont_coarse",
        "grid_lat_coarse",
        "grid_lon_coarse",
        "area_coarse",
        "dq3dt_mp_coarse",
        "dq3dt_pbl_coarse",
        "dq3dt_shal_conv_coarse",
        "dt3dt_lw_coarse",
        "dt3dt_mp_coarse",
        "dt3dt_pbl_coarse",
        "dt3dt_shal_conv_coarse",
        "dt3dt_sw_coarse",
    ],
}
N = 16
SELECTORS = {
    "restart": dict(tile=[0], time=[0, 1, 2], grid_xt=slice(0, N), grid_yt=slice(0, N)),
    "atmos_15min_coarse_ave": dict(
        tile=[0],
        time=[0, 1],
        grid_xt_coarse=slice(0, N),
        grid_yt_coarse=slice(0, N),
        grid_x_coarse=slice(0, N + 1),
        grid_y_coarse=slice(0, N + 1),
    ),
    "gfsphysics_15min_coarse": dict(
        tile=[0],
        time=[0, 1],
        grid_xt_coarse=slice(0, N),
        grid_yt_coarse=slice(0, N),
        grid_x_coarse=slice(0, N + 1),
        grid_y_coarse=slice(0, N + 1),
    ),
}
EXPECTED_VARIABLES = [
    "T",
    "t_dt_fv_sat_adj_coarse",
    "t_dt_nudge_coarse",
    "t_dt_phys_coarse",
    "delp",
    "vulcan_omega_coarse",
    "sphum",
    "qv_dt_fv_sat_adj_coarse",
    "qv_dt_phys_coarse",
    "sphum_vulcan_omega_coarse",
    "T_vulcan_omega_coarse",
    "eddy_flux_vulcan_omega_temp",
    "eddy_flux_vulcan_omega_sphum",
    "T_storage",
    "sphum_storage",
    "dq3dt_mp_coarse",
    "dq3dt_pbl_coarse",
    "dq3dt_shal_conv_coarse",
    "dt3dt_lw_coarse",
    "dt3dt_mp_coarse",
    "dt3dt_pbl_coarse",
    "dt3dt_shal_conv_coarse",
    "dt3dt_sw_coarse",
]


ranges = {
    # Need to use a small range here to avoid SEGFAULTS in the mappm
    # if delp varies to much then the mean pressures may lie completely out of bounds
    # an individual column
    "delp": synth.Range(0.99, 1.01)
}


def open_schema(path_relative_to_file):
    path = Path(__file__)
    abspath = path.parent / path_relative_to_file
    with open(abspath) as f:
        return synth.generate(synth.load(f), ranges)


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
def test_run(tmpdir):
    paths = {}
    for name, schema_file in SCHEMAS.items():
        ds = open_schema(schema_file)
        ds = safe.get_variables(ds, VARIABLES[name])
        ds = ds.isel(SELECTORS[name])  # Subset for faster testing
        path = str(tmpdir.join(f"{name}.zarr"))
        paths[name] = path
        ds.to_zarr(path, mode="w")

    output_path = tmpdir.join("out")
    run(
        paths["restart"],
        paths["atmos_15min_coarse_ave"],
        paths["gfsphysics_15min_coarse"],
        output_path,
    )

    ds = xr.open_mfdataset(f"{output_path}/*.nc", combine="by_coords")

    for variable in EXPECTED_VARIABLES:
        assert variable in ds
        assert "long_name" in ds[variable].attrs
        assert "units" in ds[variable].attrs


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
