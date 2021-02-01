#!/bin/bash

set -eo pipefail

CONFIG="$1"
RUNDIR="$2"
RUNFILE="$3"

# initialize the output directory so the logs.txt can be created before the run
# directory writing stage finishes
mkdir -p "$RUNDIR"

# Setting this variable so we can use mpich with > 6 ranks on
# a single node. The alternative is to set --shm-size in the docker
# run command.  Without it, C48 runs with 24 ranks crash with a "bus error."
export MPIR_CVAR_CH3_NOLOCAL=1

write_run_directory "$CONFIG" "$RUNDIR"
cp "$RUNFILE" "$RUNDIR/runfile.py"
cd "$RUNDIR"
NUM_PROC=$(yq '.namelist.fv_core_nml.layout | .[0] *.[1] * 6' "$CONFIG")
mpirun -n "$NUM_PROC" python3 "$RUNFILE" |& tee -a "$RUNDIR/logs.txt"
