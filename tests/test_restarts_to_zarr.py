from fv3net.pipelines.restarts_to_zarr.funcs import run

import pytest


@pytest.mark.regression
def test_restarts_to_zarr(tmpdir):
    output = tmpdir.join("out")
    known_args = [
        "gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files",
        output,
    ]
    pipeline_args = [
        "--runner",
        "Direct",
        "--num_workers",
        1,
        "--n-steps",
        1,
    ]
    run(known_args, pipeline_args)
