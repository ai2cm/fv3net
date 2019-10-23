import logging

from google.cloud.storage import Client

from coarseflow.pipeline import run
from coarseflow.file_lister import GCSLister

"""
Run command used for Dataflow

python coarseflow_main.py \
    --job_name coarseflow-test-andrep \
    --project vcm-ml \
    --region us-central1 \
    --runner DataflowRunner \
    --setup_file ./setup.py \
    --temp_location gs://vcm-ml-data/test_dataflow \
    --max_num_workers 4 \
    --disk_size_gb 80 \
    --extra_package /home/andrep/repos/fv3net/scratch/coarsen-dataflow.tar \
    --type_check_strictness 'ALL_REQUIRED'
"""

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run(GCSLister(Client(), 'vcm-ml-data'),
        prefix='test_dataflow',
        file_extension='tar')
