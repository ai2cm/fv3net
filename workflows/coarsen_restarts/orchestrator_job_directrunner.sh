#!/bin/bash

INPUT_PATH=$1
GRIDSPEC_PATH=$2
OUTPUT_PATH=$3
SRC_RESOLUTION=$4
TARGET_RESOLUTION=$5

python -m fv3net.pipelines.coarsen_restarts\
    $INPUT_PATH \
    $OUTPUT_PATH \
    $GRIDSPEC_PATH \
    $SRC_RESOLUTION \
    $TARGET_RESOLUTION \
    --no-target-subdir \
    --runner DirectRunner