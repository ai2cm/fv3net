#!/bin/sh

INPUT_PATH=$1
OUTPUT_PATH=$2
GCS_GRIDSPEC_PATH=$3
SRC_RESOLUTION=$4
TARGET_RESOLUTION=$5

python -m fv3net.pipelines.coarsen_restarts\
    $INPUT_PATH \
    $OUTPUT_PATH \
    $GCS_GRIDSPEC_PATH \
    $SRC_RESOLUTION \
    $TARGET_RESOLUTION \
    --no-target-subdir \
    --runner DataflowRunner \
    --job_name coarsen-restarts-$(whoami) \
    --project vcm-ml \
    --region us-central1 \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers 3 \
    --max_num_workers 50 \
    --disk_size_gb 50 \
    --worker_machine_type n1-highmem-4 \
    --setup_file ./setup.py \
    --extra_package external/vcm/dist/vcm-0.1.0.tar.gz \
    --extra_package external/vcm/external/mappm/dist/mappm-0.0.0.tar.gz