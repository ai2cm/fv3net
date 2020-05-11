#!/bin/bash

set -e

MODEL_CONFIGURATION=$1

STAMP=$(uuid | head -c 6)
ONE_STEP_DATA=gs://vcm-ml-experiments/2020-04-22-advisory-council/${MODEL_CONFIGURATION}/one_step_run
HI_RES_DIAGS=gs://vcm-ml-data/orchestration-testing/shield-coarsened-diags-2019-12-04
TIMESTEPS_FILE=./train_and_test_times.json
DIAGS_CONFIG=./workflows/one_step_diags/one_step_diags_ex_config.yaml
NETCDF_OUTPUT=gs://vcm-ml-scratch/brianh/one-step-diags-testing/${MODEL_CONFIGURATION}-${STAMP}
REPORT_DIRECTORY=gs://vcm-ml-public/2020-04-22-advisory-council/${MODEL_CONFIGURATION}-test/one-step-diagnostics

LOGDIR="./logs"
if [ ! -d $LOGDIR ]; then
  mkdir $LOGDIR
fi
LOGFILE="${LOGDIR}/one-step-diags-${STAMP}.log"
exec > >(tee ${LOGFILE}) 2>&1

export PYTHONPATH="./workflows/one_step_diags/"

CMD="python -m one_step_diags \
$ONE_STEP_DATA \
$HI_RES_DIAGS \
$TIMESTEPS_FILE \
$NETCDF_OUTPUT \
--report_directory $REPORT_DIRECTORY \
--diags_config $DIAGS_CONFIG \
--n_sample_inits 2 \
--runner DirectRunner
"
    
echo "Running command:"
echo ${CMD}

$CMD
