#!/bin/bash

set -e

NUDGING_ROOT=$1
REFERENCE_RESTART=$2
REFERENCE_PHYSICS=$3
OUTPUT=$4

python make_nudging_animations.py $NUDGING_ROOT $REFERENCE_RESTART $REFERENCE_PHYSICS /tmp
gsutil cp /tmp/*.mp4 $OUTPUT