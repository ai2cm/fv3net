#!/bin/bash

set -e

FV3CONFIG=test_fv3config.yaml
RUNFILE=../prognostic_c48_run/nudge_to_obs/runfile.py
CHUNKS=test_chunks.yaml
FV3GFS_IMAGE="us.gcr.io/vcm-ml/fv3gfs-python:v0.4.3"
POST_PROCESS_IMAGE="us.gcr.io/vcm-ml/post_process_run:48814ed3752ac507c0e19fa001eef691c9f351e8"
OUTPUT_URL="gs://vcm-ml-scratch/oliwm/test-fv3gfs-segments/v1"

argo submit run-fv3gfs.yaml -p fv3config="$(cat $FV3CONFIG)" -p runfile="$(cat $RUNFILE)" -p chunks="$(cat $CHUNKS)" -p fv3gfs-image=${FV3GFS_IMAGE} -p post-process-image=${POST_PROCESS_IMAGE} -p output-url=${OUTPUT_URL} -p submission-count=2


#argo submit run-fv3gfs.yaml \
#    -p fv3config="$(cat $FV3CONFIG)" \ 
#    -p runfile="$(cat $RUNFILE)" \ 
#    -p chunks="$(cat $CHUNKS)" \ 
#    -p fv3gfs-image=${FV3GFS_IMAGE} \ 
#    -p output-url=${OUTPUT_URL} \ 
#    -p submission-count=2
