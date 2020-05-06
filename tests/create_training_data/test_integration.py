import os
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
    '''
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.

    Credit: https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    '''
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

    big_zarr = schema.generate()
    pipeline_args = []
    names = get_config({})

    diag_c48_path = "gs://vcm-ml-data/testing-noah/2020-04-18/25b5ec1a1b8a9524d2a0211985aa95219747b3c6/coarsen_diagnostics/"
    output_dir = "./out"

    run(big_zarr, diag_c48_path, output_dir, pipeline_args, names, timesteps)
