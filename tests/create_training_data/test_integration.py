import os
import xarray as xr
import synth
from distutils import dir_util
from fv3net.pipelines.create_training_data.config import get_config
from fv3net.pipelines.create_training_data.pipeline import run

import pytest


timesteps = {
    "train": [
        ["20160801.003000", "20160801.004500"],
        ["20160801.001500", "20160801.003000"],
    ],
    "test": [
        ["20160801.011500", "20160801.013000"],
        ["20160801.010000", "20160801.011500"],
    ],
}


@pytest.fixture
def datadir(tmpdir, request):
    """
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.

    Credit: https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir


@pytest.mark.regression()
def test_create_training_data_regression(datadir):
    path = datadir.join("schema.json")
    with open(str(path)) as f:
        schema = synth.load(f)

    path = datadir.join("diag.json")
    with open(str(path)) as f:
        diag_schema = synth.load(f)

    big_zarr = schema.generate()
    ds_diag = diag_schema.generate()
    # need to decode the time coordinate.
    ds_diag_decoded = xr.decode_cf(ds_diag)

    pipeline_args = []
    names = get_config({})

    output_dir = "./out"

    # import fsspec
    # import xarray as xr

    # diag_c48_path = "gs://vcm-ml-data/testing-noah/2020-04-18/25b5ec1a1b8a9524d2a0211985aa95219747b3c6/coarsen_diagnostics/"
    # COARSENED_DIAGS_ZARR_NAME = "gfsphysics_15min_coarse.zarr"
    # full_zarr_path = os.path.join(diag_c48_path, COARSENED_DIAGS_ZARR_NAME)
    # mapper = fsspec.get_mapper(full_zarr_path)
    # ds_diag = xr.open_zarr(mapper, consolidated=True)


    run(big_zarr, ds_diag_decoded, output_dir, pipeline_args, names, timesteps)
