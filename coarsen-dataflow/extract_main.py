import logging

from google.cloud.storage import Client

from coarseflow.pipeline import run
from coarseflow.file_lister import GCSLister

"""
Run command used for Dataflow

Adjust the following to meet job needs
------------------
num_workers: number of workers to start with
max_num_workers: autoscaling limit

Running defaults machine-type the 1-core 3.75 GB memory workers

python extract_main.py \
    --job_name continue-extract-all-tars-andrep \
    --project vcm-ml \
    --region us-central1 \
    --runner DataflowRunner \
    --setup_file ./setup.py \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers 60 \
    --max_num_workers 60 \
    --disk_size_gb 80 \
    --type_check_strictness 'ALL_REQUIRED' 
"""

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run(GCSLister(Client(), 'vcm-ml-data'),
        prefix='2019-10-05-X-SHiELD-C3072-to-C384-re-uploaded-restart-data',
        file_extension='tar',
        output_prefix='2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted')

    