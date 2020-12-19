#!/bin/bash

set -eo pipefail

CONFIG="$1"
RUNDIR="$2"
RUNFILE="$3"

# initialize the output directory so the logs.txt can be created before the run
# directory writing stage finishes
mkdir -p "$RUNDIR"


write_run_directory "$CONFIG" "$RUNDIR"
cp "$RUNFILE" "$RUNDIR"
cd "$RUNDIR"
NUM_PROC=$(yq '.namelist.fv_core_nml.layout | .[0] *.[1] * 6' "$CONFIG")
mpirun -n "$NUM_PROC" python3 "$RUNFILE" |& tee -a "$RUNDIR/logs.txt"