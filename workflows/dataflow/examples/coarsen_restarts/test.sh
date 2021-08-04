#!/bin/sh

GCS_SRC="gs://vcm-ml-scratch/annak/test-pire-data/" 
GCS_GRIDSPEC="gs://vcm-ml-data/2020-01-06-C384-grid-spec-with-area-dx-dy" 
SRC_RESOLUTION=384
TARGET_RESOLUTION=48
GCS_DST="gs://vcm-ml-scratch/annak/2021-08-04-correct-dtype-pire-coarsened-restarts"  #gs://vcm-ml-intermediate/2021-07-28-PIRE-c48-restarts-post-spinup"

python -m fv3net.pipelines.coarsen_restarts\
    $GCS_SRC \
    $GCS_GRIDSPEC \
    $SRC_RESOLUTION \
    $TARGET_RESOLUTION \
    $GCS_DST \
    --type_check_strictness ALL_REQUIRED \
    --runner DirectRunner 
