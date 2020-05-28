import synth
import os
import pytest
import sys
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


@pytest.fixture()
def restart_dir(tmpdir):
    time = "20160101.000000"
    output = tmpdir.mkdir(time)
    tmpdir.mkdir("output").mkdir(time)
    restarts = synth.generate_restart_data(n=384)
    save_restarts(restarts, output, time)
    return tmpdir


def test_regression_coarsen_restarts(restart_dir):
    grid_spec_path = "gs://vcm-ml-data/2020-01-06-C384-grid-spec-with-area-dx-dy/"
    src_path = str(restart_dir)
    in_res = "384"
    out_res = "48"
    dest = str(restart_dir.join("output"))

    main(
        [src_path, grid_spec_path, in_res, out_res, dest,]
    )
