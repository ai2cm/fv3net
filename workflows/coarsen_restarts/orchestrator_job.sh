#!/bin/sh

GCS_SRC=$1
GCS_DST=$2
GCS_GRIDSPEC="gs://vcm-ml-data/2020-01-06-C384-grid-spec-with-area-dx-dy"

SRC_RESOLUTION=$3
TARGET_RESOLUTION=$4

python -m fv3net.pipelines.coarsen_restarts\
    --gcs-src-dir $GCS_SRC \
    --gcs-dst-dir $GCS_DST \
    --gcs-grid-spec-path $GCS_GRIDSPEC \
    --source-resolution $SRC_RESOLUTION \
    --target-resolution $TARGET_RESOLUTION \
    --add-target-subdir False \
    --runner DataflowRunner \
    --job_name coarsen-restarts-$(whoami) \
    --project vcm-ml \
    --region us-central1 \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers 10 \
    --max_num_workers 50 \
    --disk_size_gb 50 \
    --worker_machine_type n1-highmem-4 \
    --setup_file ./setup.py \
    --extra_package external/vcm/dist/vcm-0.1.0.tar.gz \
    --extra_package external/vcm/external/mappm/dist/mappm-0.0.0.tar.gz