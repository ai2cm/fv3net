import synth
from pathlib import Path
import xarray as xr
import logging
from vcm import safe
import cftime
import datetime

from budget.data import shift
from budget.pipeline import run

ranges = {
    # Need to use a small range here to avoid SEGFAULTS in the mappm
    # if delp varies to much then the mean pressures may lie completely out of bounds
    # an individual column
    "delp": synth.Range(0.99, 1.01)
}


def open_schema(localpath):
    path = Path(__file__)
    diag_path = path.parent / localpath
    with open(diag_path) as f:
        return synth.generate(synth.load(f), ranges)


def test_run(tmpdir):

    logging.basicConfig(level=logging.INFO)

    variables = [
        "t_dt_gfdlmp_coarse",
        "t_dt_nudge_coarse",
        "t_dt_phys_coarse",
        "qv_dt_gfdlmp_coarse",
        "qv_dt_phys_coarse",
        "eddy_flux_omega_sphum",
        "eddy_flux_omega_temp",
        "grid_lat_coarse",
        "grid_latt_coarse",
        "grid_lon_coarse",
        "grid_lont_coarse",
        "omega_coarse",
        "area_coarse",
    ]

    selectors = dict(tile=slice(0, 1), time=slice(0, 3))

    diag_schema = safe.get_variables(open_schema("diag.json"), variables).isel(
        selectors
    )
    restart = open_schema("restart.json").isel(selectors)

    diag_path = str(tmpdir.join("diag.zarr"))
    restart_path = str(tmpdir.join("restart.zarr"))
    output_path = str(tmpdir.join("out"))

    diag_schema.to_zarr(diag_path, mode="w")
    restart.to_zarr(restart_path, mode="w")

    run(restart_path, diag_path, output_path)


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
