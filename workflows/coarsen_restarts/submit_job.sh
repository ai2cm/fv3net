#!/bin/sh

GCS_SRC="gs://vcm-ml-data/2019-12-02-40-day-X-SHiELD-simulation-C384-restart-files"
GCS_DST="gs://vcm-ml-data/test_andrep/coarsen_test"
GCS_GRIDSPEC="gs://vcm-ml-data/2020-01-06-C384-grid-spec-with-area-dx-dy"

SRC_RESOLUTION=384
TARGET_RESOLUTION=48

python -m fv3net.pipelines.coarsen_timesteps\
    --gcs-src-dir $GCS_SRC \
    --gcs-dst-dir $GCS_DST \
    --gcs-grid-spec-path $GCS_GRIDSPEC \
    --source-resolution $SRC_RESOLUTION \
    --target_resolution $TARGET_RESOLUTION \
    --runner DataflowRunner \
    --job_name test-coarsen-restarts-$(whoami) \
    --project vcm-ml \
    --region us-central1 \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers 1 \
    --max_num_workers 1 \
    --disk_size_gb 50 \
    --worker_machine_type n1-standard-2 \
    --setup_file ./setup.py \
    --extra_package external/vcm/dist/vcm-0.1.0.tar.gz \
    --extra_package external/vcm/external/mappm/dist/mappm-0.0.0.tar.gz