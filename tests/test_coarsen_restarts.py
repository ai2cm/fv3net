import synth
import os
import pytest
from fv3net.pipelines.coarsen_restarts.pipeline import main


def save_restarts(restarts, outdir, time):

    OUTPUT_CATEGORY_NAMES = {
        "fv_core.res": "fv_core_coarse.res",
        "fv_srf_wnd.res": "fv_srf_wnd_coarse.res",
        "fv_tracer.res": "fv_tracer_coarse.res",
        "sfc_data": "sfc_data_coarse",
    }

    for category, tiles in restarts.items():
        for tile, dataset in tiles.items():
            out_category = OUTPUT_CATEGORY_NAMES[category]
            filename = os.path.join(outdir, f"{time}.{out_category}.tile{tile}.nc")
            dataset.to_netcdf(filename)


def _grid_spec(datadir, nx):
    scaling_factor = 384 / nx
    ranges = {
        "dx": synth.Range(20000 * scaling_factor, 28800 * scaling_factor),
        "dy": synth.Range(20000 * scaling_factor, 28800 * scaling_factor),
        "area": synth.Range(
            3.6205933e08 * scaling_factor ** 2, 8.3428736e08 * scaling_factor ** 2
        ),
    }
    path = str(datadir.join("grid_spec.json"))
    with open(path) as f:
        schema = synth.load(f)

    ds = synth.generate(schema, ranges)
    subset = ds.isel(
        grid_xt=slice(nx), grid_yt=slice(nx), grid_x=slice(nx + 1), grid_y=slice(nx + 1)
    )

    return subset


@pytest.fixture()
def restart_dir(tmpdir, datadir):

    nx = 48

    time = "20160101.000000"
    output = tmpdir.mkdir(time)
    tmpdir.mkdir("grid_spec")
    tmpdir.mkdir("output").mkdir(time)
    restarts = synth.generate_restart_data(nx=nx)
    save_restarts(restarts, output, time)

    grid_spec = _grid_spec(datadir, nx)
    for i in range(6):
        path = str(tmpdir.join(f"grid_spec/grid_spec.tile{i+1}.nc"))
        grid_spec.to_netcdf(path)

    return tmpdir


@pytest.mark.regression
def test_regression_coarsen_restarts(restart_dir):
    grid_spec_path = str(restart_dir.join("grid_spec"))
    src_path = str(restart_dir)
    in_res = "48"
    out_res = "6"
    dest = str(restart_dir.join("output"))

    main([src_path, grid_spec_path, in_res, out_res, dest])
