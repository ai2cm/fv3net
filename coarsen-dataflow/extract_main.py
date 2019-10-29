import logging

from google.cloud.storage import Client

from coarseflow.pipeline import run
from coarseflow.file_lister import GCSLister

"""
Run command used for Dataflow

Likely adjustments
------------------
num_workers: number of workers to start with
max_num_workers: autoscaling limit

Running defaults to the 1-core 3.75 GB memory workers

python extract_main.py \
    --job_name coarseflow-test-andrep \
    --project vcm-ml \
    --region us-central1 \
    --runner DataflowRunner \
    --setup_file ./setup.py \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers 4 \
    --max_num_workers 4 \
    --disk_size_gb 80 \
    --type_check_strictness 'ALL_REQUIRED' \
    --service_account_email andre-vm-sa@vcm-ml.iam.gserviceaccount.com
"""

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run(GCSLister(Client(), 'vcm-ml-data'),
        prefix='test_dataflow',
        file_extension='tar')

    