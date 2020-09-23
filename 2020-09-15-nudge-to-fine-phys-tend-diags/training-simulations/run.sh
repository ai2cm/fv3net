#!/bin/bash

set -e

FV3NET_HASH=b8816a17832746b7cb4185a56e22c9baec5581bb
FV3GFS_WRAPPER_HASH=46b1d8742ee220ebeb14942e0a0f43da963cc0cf

NUDGING_CONFIG="$(< $1)"
OUTPUT_TIMES="$(< $2)"
EXPERIMENT=$3

argo submit --from workflowtemplate/nudging \
    -p fv3net-image=us.gcr.io/vcm-ml/fv3net:$FV3NET_HASH \
    -p fv3gfs-image=us.gcr.io/vcm-ml/fv3gfs-wrapper:$FV3GFS_WRAPPER_HASH \
    -p post-process-image=us.gcr.io/vcm-ml/post_process_run:$FV3NET_HASH \
    -p nudging-config="$NUDGING_CONFIG" \
    -p segment-count=19 \
    -p output-url=gs://vcm-ml-experiments/2020-09-16-$EXPERIMENT \
    -p times="$OUTPUT_TIMES"
