import subprocess

import pytest


_test_script = """

NUM_WORKERS=1

python -m fv3net.pipelines.restarts_to_zarr  \
    gs://vcm-ml-data/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files \
    {output} \
    --runner Direct \
    --num_workers $NUM_WORKERS \
    --n-steps 1  \

"""

@pytest.mark.regression
def test_restarts_to_zarr(tmpdir):
    output = tmpdir.join("out")
    subprocess.check_call(['bash', '-c', _test_script.format(output=output)])
