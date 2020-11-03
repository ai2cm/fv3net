import synth
from vcm import safe
from pathlib import Path
import pytest


def open_schema(path_relative_to_file):
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


@pytest.fixture()
def data_dirs(tmpdir):

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

    # use a small tile for much faster testing
    n = 48

    diag_selectors = dict(
        tile=[0], time=[0, 1], grid_xt_coarse=slice(0, n), grid_yt_coarse=slice(0, n)
    )

    restart_selectors = dict(
        tile=[0], time=[0, 1, 2], grid_xt=slice(0, n), grid_yt=slice(0, n)
    )

    diag_schema = safe.get_variables(open_schema("diag.json"), variables).isel(
        diag_selectors
    )
    restart = open_schema("restart.json").isel(restart_selectors)

    diag_path = str(tmpdir.join("diag.zarr"))
    restart_path = str(tmpdir.join("restart.zarr"))

    diag_path = str(tmpdir.join("diag.zarr"))
    restart_path = str(tmpdir.join("restart.zarr"))

    diag_schema.to_zarr(diag_path, mode="w")
    restart.to_zarr(restart_path, mode="w")

    return diag_path, restart_path, expected_variables

