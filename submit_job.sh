#!/bin/bash

python -m $1  \
    --job_name test-$(uuid) \
    --project vcm-ml \
    --region us-central1 \
    --runner DataFlow \
    --setup_file ./setup.py \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers 1 \
    --max_num_workers 1 \
    --disk_size_gb 50 \
    --worker_machine_type n1-standard-2 \
    --extra_package external/vcm/dist/vcm-0.1.0.tar.gz
