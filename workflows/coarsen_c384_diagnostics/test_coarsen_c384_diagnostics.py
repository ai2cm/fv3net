import os
import pathlib
import xarray as xr
import numpy as np

from coarsen_c384_diagnostics import coarsen_c384_diagnostics, _create_arg_parser

workflow_path = pathlib.Path(__file__).parent.absolute()


def grid_ds():
    area = xr.DataArray(np.ones((6, 384, 384)), dims=["tile", "grid_xt", "grid_yt"])
    return xr.Dataset({"area": area})


def c384_ds():
    pratesfc = xr.DataArray(
        np.ones((10, 6, 384, 384)), dims=["time", "tile", "grid_xt", "grid_yt"]
    )
    shtflsfc = xr.DataArray(
        np.ones((10, 6, 384, 384)), dims=["time", "tile", "grid_xt", "grid_yt"]
    )
    return xr.Dataset({"PRATEsfc_coarse": pratesfc, "SHTFLsfc_coarse": shtflsfc})


def coarsen_workflow_args(input_path, grid_path, output_path):
    return [
        input_path,
        os.path.join(workflow_path, "coarsen-c384-diagnostics.yml"),
        "--grid_spec",
        grid_path,
        output_path,
    ]


def test_coarsen_c384_diagnostics(tmpdir):
    input_path = str(tmpdir.join("physics_diags.zarr"))
    grid_path = str(tmpdir.join("grid.zarr"))
    output_path = str(tmpdir.join("output"))
    grid_ds().to_zarr(grid_path)
    c384_ds().to_zarr(input_path)
    parser = _create_arg_parser()
    args = parser.parse_args(coarsen_workflow_args(input_path, grid_path, output_path))
    coarsen_c384_diagnostics(args)
    output_ds = xr.open_zarr(os.path.join(output_path, "physics_diags.zarr"))
    assert dict(output_ds.sizes) == {
        "time": 10,
        "tile": 6,
        "grid_xt": 48,
        "grid_yt": 48,
    }
