#!/bin/sh

INPUT_PATH=$1
OUTPUT_PATH=$2
GCS_GRIDSPEC_PATH=$3
SRC_RESOLUTION=$4
TARGET_RESOLUTION=$5

python -m fv3net.pipelines.coarsen_restarts\
    --gcs-src-dir $INPUT_PATH \
    --gcs-dst-dir $OUTPUT_PATH \
    --gcs-grid-spec-path $GCS_GRIDSPEC_PATH \
    --source-resolution $SRC_RESOLUTION \
    --target-resolution $TARGET_RESOLUTION \
    --no-target-subdir \
    --runner DirectRunner