#!/bin/bash

set -eo pipefail

CONFIG="$1"
RUNDIR="$2"
RUNFILE="$3"

# initialize the output directory so the logs.txt can be created before the run
# directory writing stage finishes
mkdir -p "$RUNDIR"

python -m fv3config.fv3run._native_main \
    "[[\"$CONFIG\", \"$RUNDIR\"], {\"runfile\": \"$RUNFILE\", \"capture_output\": false}]" \
    |& tee -a  "$RUNDIR/logs.txt"