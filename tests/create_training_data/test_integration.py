from zarr_to_test_schema import load
from fv3net.pipelines.create_training_data.config import get_config
from fv3net.pipelines.create_training_data.pipeline import run


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


def test_create_training_data_regression():
    with open("schema.json") as f:
        schema = load(f)

    big_zarr = schema.generate()
    pipeline_args = []
    names = get_config({})

    diag_c48_path = "gs://vcm-ml-data/testing-noah/2020-04-18/25b5ec1a1b8a9524d2a0211985aa95219747b3c6/coarsen_diagnostics/"
    output_dir = "./out"

    run(big_zarr, diag_c48_path, output_dir, pipeline_args, names, timesteps)
