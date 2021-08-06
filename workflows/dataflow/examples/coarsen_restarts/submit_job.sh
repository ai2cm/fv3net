#!/bin/sh

GCS_SRC="gs://vcm-ml-raw-flexible-retention/2021-07-19-PIRE/C3072-to-C384-restart-files" 
GCS_GRIDSPEC="gs://vcm-ml-data/2020-01-06-C384-grid-spec-with-area-dx-dy" 
SRC_RESOLUTION=384
TARGET_RESOLUTION=48
GCS_DST="gs://vcm-ml-intermediate/2021-08-06-PIRE-c48-restarts-post-spinup"

python -m fv3net.pipelines.coarsen_restarts\
    $GCS_SRC \
    $GCS_GRIDSPEC \
    $SRC_RESOLUTION \
    $TARGET_RESOLUTION \
    $GCS_DST \
    --coarsen-agrid-winds \
    --runner DataflowRunner \
    --job_name coarsen-restarts-pire-full-post-spinup \
    --project vcm-ml \
    --region us-central1 \
    --temp_location gs://vcm-ml-scratch/tmp_dataflow \
    --num_workers 3 \
    --max_num_workers 100 \
    --disk_size_gb 50 \
    --worker_machine_type n1-standard-4 \
    --setup_file /home/AnnaK/fv3net/workflows/dataflow/setup.py \
    --extra_package /home/AnnaK/fv3net/external/vcm/dist/vcm-0.1.0.tar.gz \
    --extra_package /home/AnnaK/fv3net/external/vcm/external/mappm/dist/mappm-0.1.0.tar.gz