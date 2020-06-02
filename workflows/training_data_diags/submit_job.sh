#!/bin/bash

set -e

CONFIG_FILE=$1

LOGDIR="./logs"
if [ ! -d $LOGDIR ]; then
  mkdir $LOGDIR
fi
LOGFILE="${LOGDIR}/one-step-diags-${STAMP}.log"
exec > >(tee ${LOGFILE}) 2>&1

export PYTHONPATH="./workflows/training_data_diags/"

CMD="python -m training_data_diags $CONFIG_FILE"
    
echo "Running command:"
echo ${CMD}

$CMD
