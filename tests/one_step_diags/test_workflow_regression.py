from one_step_diags import config
from one_step_diags.config import (
    INIT_TIME_DIM,
    FORECAST_TIME_DIM,
    OUTPUT_NC_FILENAME,
    ZARR_STEP_DIM,
)
from one_step_diags.pipeline import run
from vcm.safe import get_variables
import synth
import xarray as xr
import fsspec
import pytest
import os
import logging

logger = logging.getLogger(__name__)

timesteps = [
    ["20160811.090000", "20160811.091500"],
    ["20160828.060000", "20160828.061500"],
]


@pytest.mark.regression()
def test_one_step_diags_regression(datadir):

    output_dir = str(datadir.join("out"))
    one_step_zarrpath = os.path.join(output_dir, "one_step.zarr")
    hi_res_diags_zarrpath = os.path.join(output_dir, "hi_res_diags.zarr")
    output_nc_path = os.path.join(output_dir, OUTPUT_NC_FILENAME)

    default_config = {key: getattr(config, key) for key in config.__all__}

    path = datadir.join("one_step_zarr_schema.json")
    with open(str(path)) as f:
        one_step_schema = synth.load(f)

    path = datadir.join("hi_res_diags_zarr_schema.json")
    with open(str(path)) as f:
        hi_res_diags_schema = synth.load(f)

    path = datadir.join("one_step_diags_schema.json")
    with open(str(path)) as f:
        one_step_diags_schema = synth.load(f)

    ranges = {}

    one_step_dataset = synth.generate(one_step_schema, ranges=ranges)
    one_step_dataset.to_zarr(one_step_zarrpath, consolidated=True)

    grid = (
        get_variables(one_step_dataset, default_config["GRID_VARS"])
        .isel({INIT_TIME_DIM: 0, FORECAST_TIME_DIM: 0, ZARR_STEP_DIM: 0})
        .drop([ZARR_STEP_DIM, INIT_TIME_DIM, FORECAST_TIME_DIM])
    )

    hi_res_diags_dataset = synth.generate(hi_res_diags_schema, ranges=ranges)
    # need to decode the time coordinate.
    hi_res_diags_dataset = xr.decode_cf(hi_res_diags_dataset)
    hi_res_diags_dataset.to_zarr(hi_res_diags_zarrpath, consolidated=True)

    pipeline_args = []

    run(
        pipeline_args,
        one_step_zarrpath,
        timesteps,
        hi_res_diags_zarrpath,
        default_config,
        grid,
        output_nc_path,
    )

    with fsspec.open(output_nc_path, "rb") as f:
        pipeline_output_dataset = xr.open_dataset(f).load()

    output_schema = synth.read_schema_from_dataset(pipeline_output_dataset)

    print(output_schema)
    print(one_step_diags_schema)

    assert output_schema == one_step_diags_schema
