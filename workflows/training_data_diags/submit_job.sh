#!/bin/bash

set -e

CONFIG_FILE=$1
echo ${CONFIG_FILE}
# CONFIG_FILE="workflows/training_data_diags/triad_experiments.yml"
OUTPUT_PATH="gs://vcm-ml-scratch/brianh/training-data-diagnostics"


LOGDIR="./logs"
if [ ! -d $LOGDIR ]; then
  mkdir $LOGDIR
fi
LOGFILE="${LOGDIR}/training_data-diags-$(uuid | head -c 6).log"
exec > >(tee ${LOGFILE}) 2>&1

export PYTHONPATH="./workflows/training_data_diags/"

CMD="python ./workflows/training_data_diags/compute_diags.py ${CONFIG_FILE} ${OUTPUT_PATH}"
    
echo "Running command:"
echo ${CMD}

$CMD
