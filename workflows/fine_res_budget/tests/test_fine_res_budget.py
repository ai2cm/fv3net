import synth
from pathlib import Path
import xarray as xr
from vcm import safe
import sys

from budget.pipeline import run


ranges = {
    "delp": synth.Range(1, 10)
}

def open_schema(localpath):
    path = Path(__file__)
    diag_path = path.parent / localpath
    with open(diag_path) as f:
        return synth.generate(synth.load(f), ranges)


def test_run(tmpdir):

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
        "area_coarse"
    ]

    selectors = dict(tile=slice(0, 1), time=slice(0, 3))

    diag_schema = safe.get_variables(open_schema("diag.json"), variables).isel(selectors)
    restart = open_schema("restart.json").isel(selectors)

    diag_path = str(tmpdir.join("diag.zarr"))
    restart_path = str(tmpdir.join("restart.zarr"))
    output_path = str(tmpdir.join("out"))

    diag_schema.to_zarr(diag_path, mode="w")
    restart.to_zarr(restart_path, mode="w")

    run(restart_path, diag_path, output_path)
