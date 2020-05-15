from fv3net.pipelines.restarts_to_zarr.funcs import main

import pytest


@pytest.mark.regression
def test_restarts_to_zarr(tmpdir):
    output = tmpdir.join("out")

    url = "gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files"

    argv = [
        url,
        str(output),
        "--n-steps",
        "1",
        "--runner",
        "Direct",
        "--num_workers",
        "1",
    ]
    main(argv)
