#!/bin/bash

set -e

CONFIG="$1"
RUNDIR="$2"
RUNFILE="$3"

# make the run directory so the logs.txt can be created before write_run_directory finishes
mkdir -p "$RUNDIR"

python -m fv3config.fv3run._native_main \
    "[[\"$CONFIG\", \"$RUNDIR\"], {\"runfile\": \"$RUNFILE\", \"capture_output\": false}]" \
    |& tee -a  "$RUNDIR/logs.txt"