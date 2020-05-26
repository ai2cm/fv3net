import xarray as xr
import synth
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


@pytest.mark.regression()
def test_create_training_data_regression(datadir):

    output_dir = str(datadir.join("out"))

    path = datadir.join("schema.json")
    with open(str(path)) as f:
        schema = synth.load(f)

    path = datadir.join("diag.json")
    with open(str(path)) as f:
        diag_schema = synth.load(f)

    big_zarr = synth.generate(schema)
    ds_diag = synth.generate(diag_schema)
    # need to decode the time coordinate.
    ds_diag_decoded = xr.decode_cf(ds_diag)

    pipeline_args = []
    names = get_config({})

    run(big_zarr, ds_diag_decoded, output_dir, pipeline_args, names, timesteps)
