#!/bin/bash

set -e

STAMP=$(uuid | head -c 6)
MODEL_VERSION=deep-off
ONE_STEP_DATA=gs://vcm-ml-experiments/2020-04-22-advisory-council/${MODEL_VERSION}/one_step_run
HI_RES_DIAGS=gs://vcm-ml-data/orchestration-testing/shield-coarsened-diags-2019-12-04
TIMESTEPS_FILE=./train_and_test_times.json
DIAGS_CONFIG=./workflows/one_step_diags/one_step_diags_ex_config.yaml
NETCDF_OUTPUT=gs://vcm-ml-scratch/brianh/one-step-diags-testing/${MODEL_VERSION}-${STAMP}
REPORT_DIRECTORY=gs://vcm-ml-public/2020-04-22-advisory-council/${MODEL_VERSION}-test/one-step-diagnostics

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
--n_sample_inits 48 \
--runner DataflowRunner \
--job_name one-step-diags-${USER}-$(uuid | head -c 7) \
--project vcm-ml \
--region us-central1 \
--temp_location gs://vcm-ml-data/tmp_dataflow \
--num_workers 4 \
--max_num_workers 30 \
--disk_size_gb 100 \
--worker_machine_type n2-highmem-4 \
--setup_file ./setup.py \
--extra_package ./external/report/dist/report-0.1.0.tar.gz \
--extra_package ./workflows/one_step_diags/dist/one_step_diags-0.1.0.tar.gz
"
    
echo "Running command:"
echo ${CMD}

$CMD
