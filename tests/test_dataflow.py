import subprocess
import pytest

submission = """
bash -x dataflow.sh submit $(pwd)/tests/simple_dataflow.py  \
    --save_main_session \
    --job_name test-$(uuid) \
    --project vcm-ml \
    --region us-central1 \
    --runner DataFlow \
    --temp_location gs://vcm-ml-scratch/tmp_dataflow \
    --num_workers 1 \
    --autoscaling_algorithm=NONE \
    --worker_machine_type n1-standard-1 \
    --disk_size_gb 30
"""


@pytest.mark.regression()
def test_submit_dataflow():
    subprocess.check_call([submission], shell=True)
