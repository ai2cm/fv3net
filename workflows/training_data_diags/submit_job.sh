#!/bin/bash

set -e

CONFIG_FILE="workflows/training_data_diags/three_sources_config.yml"
OUTPUT_PATH="gs://vcm-ml-scratch/brianh/training-data-diagnostics"


LOGDIR="./logs"
if [ ! -d $LOGDIR ]; then
  mkdir $LOGDIR
fi
LOGFILE="${LOGDIR}/one-step-diags-${STAMP}.log"
exec > >(tee ${LOGFILE}) 2>&1

export PYTHONPATH="./workflows/training_data_diags/"

CMD="python -m training_data_diags ${CONFIG_FILE} ${OUTPUT_PATH}"
    
echo "Running command:"
echo ${CMD}

$CMD
