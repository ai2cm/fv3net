#!/bin/bash

set -eo pipefail

CONFIG="$1"
RUNDIR="$2"
RUNFILE="$3"

# initialize the output directory so the logs.txt can be created before the run
# directory writing stage finishes
mkdir -p "$RUNDIR"

write_run_directory "$CONFIG" "$RUNDIR"
cd "$RUNDIR"
mpirun -n 6 python3 "$RUNFILE" |& tee -a "$RUNDIR/logs.txt"