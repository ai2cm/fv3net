import synth
import xarray as xr
from vcm import safe
from pathlib import Path
import pytest

import budget.config


def open_schema(path_relative_to_file: str) -> xr.Dataset:
    path = Path(__file__)
    abspath = path.parent / path_relative_to_file
    with open(abspath) as f:
        return synth.generate(synth.load(f), ranges)


ranges = {
    # Need to use a small range here to avoid SEGFAULTS in the mappm
    # if delp varies to much then the mean pressures may lie completely out of bounds
    # an individual column
    "delp": synth.Range(0.99, 1.01)
}


@pytest.fixture(scope="session")
def data_dirs(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("input_data")

    variables = [
        "grid_lat_coarse",
        "grid_latt_coarse",
        "grid_lon_coarse",
        "grid_lont_coarse",
    ]
    # use a small tile for much faster testing
    n = 48

    diag_selectors = dict(
        tile=[0], time=[0, 1], grid_xt_coarse=slice(0, n), grid_yt_coarse=slice(0, n)
    )

    restart_selectors = dict(
        tile=[0], time=[0, 1, 2], grid_xt=slice(0, n), grid_yt=slice(0, n)
    )

    diags = safe.get_variables(
        open_schema("diag.json"), budget.config.PHYSICS_VARIABLES + variables
    ).isel(diag_selectors)
    restart = open_schema("restart.json").isel(restart_selectors)
    gfsphysics = open_schema("gfsphysics.json").isel(diag_selectors)

    diag_path = str(tmpdir.join("diag.zarr"))
    restart_path = str(tmpdir.join("restart.zarr"))
    gfsphysics_path = str(tmpdir.join("gfsphysics.zarr"))

    diags.to_zarr(diag_path, mode="w", consolidated=True)
    restart.to_zarr(restart_path, mode="w", consolidated=True)
    gfsphysics.to_zarr(gfsphysics_path, mode="w", consolidated=True)

    return diag_path, restart_path, gfsphysics_path
