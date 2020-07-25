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

from vcm import safe

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

    variables = [
        "t_dt_fv_sat_adj_coarse",
        "t_dt_nudge_coarse",
        "t_dt_phys_coarse",
        "qv_dt_fv_sat_adj_coarse",
        "qv_dt_phys_coarse",
        "eddy_flux_vulcan_omega_sphum",
        "eddy_flux_vulcan_omega_temp",
        "grid_lat_coarse",
        "grid_latt_coarse",
        "grid_lon_coarse",
        "grid_lont_coarse",
        "vulcan_omega_coarse",
        "area_coarse",
    ]

    # use a small tile for much faster testing
    n = 48

    diag_selectors = dict(
        time=[0, 1], grid_xt_coarse=slice(0, n), grid_yt_coarse=slice(0, n)
    )

    restart_selectors = dict(
        time=[0, 1, 2], grid_xt=slice(0, n), grid_yt=slice(0, n)
    )

    diag_schema = safe.get_variables(open_schema("diag.json"), variables).isel(
        diag_selectors
    )
    restart = open_schema("restart.json").isel(restart_selectors)

    diag_path = str(tmpdir.join("diag.zarr"))
    restart_path = str(tmpdir.join("restart.zarr"))
    output_path = str(tmpdir.join("out"))

    diag_schema.to_zarr(diag_path, mode="w")
    restart.to_zarr(restart_path, mode="w")

    run(restart_path, diag_path, output_path)
    run(restart_path, diag_path, ".")
    
    ds = xr.open_mfdataset(f"{output_path}/*.nc", combine="by_coords")
    
    expected_variables = [
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
    ]

    for variable in expected_variables:
        assert variable in ds


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
