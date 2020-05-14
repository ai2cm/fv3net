import subprocess
import pytest

submission = """
python tests/simple_dataflow.py  \
    --setupfile $(pwd)/setup.py \
    --save_main_session \
    --job_name test-$(uuid) \
    --project vcm-ml \
    --region us-central1 \
    --runner DataFlow \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers 1 \
    --autoscaling_algorithm=NONE \
    --worker_machine_type n1-standard-1 \
    --disk_size_gb 30
"""


@pytest.mark.regression()
def test_submit_dataflow():
    subprocess.check_call([submission], shell=True)


build_sdist = """#!/bin/bash

set -e

rm -rf dist/
python setup.py sdist

d=$(mktemp -d)
trap "rm -r \"$d\"" EXIT

python -m venv $d
source $d/bin/activate

pip install dist/*.tar.gz

python -c 'import vcm; import xarray'
"""

def test_build_sddist():
    subprocess.check_call(["bash", "-c", build_sdist])
