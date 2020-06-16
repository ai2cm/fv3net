#!/bin/bash

set -e

MODEL_CONFIGURATION=$1

STAMP=$(uuid | head -c 6)
ONE_STEP_DATA=gs://vcm-ml-experiments/2020-04-22-advisory-council/${MODEL_CONFIGURATION}/one_step_run
HI_RES_DIAGS=gs://vcm-ml-data/orchestration-testing/shield-coarsened-diags-2019-12-04
TIMESTEPS_FILE=$(pwd)/train_and_test_times.json
DIAGS_CONFIG=$(pwd)/workflows/one_step_diags/one_step_diags_ex_config.yaml
NETCDF_OUTPUT=gs://vcm-ml-scratch/brianh/one-step-diags-testing/${MODEL_CONFIGURATION}-${STAMP}

LOGDIR="./logs"
if [ ! -d $LOGDIR ]; then
  mkdir $LOGDIR
fi
LOGFILE="${LOGDIR}/one-step-diags-${STAMP}.log"
exec >>${LOGFILE} 2>&1

# workflow-specific extra packages for dataflow
(
  cd workflows/one_step_diags
  poetry build --format sdist
)
(
  cd external/gallery
  poetry build --format sdist
)
(
  cd external/report
  poetry build --format sdist
)

export PYTHONPATH="./workflows/one_step_diags/"

PYTHON_CMD=" -m one_step_diags \
$ONE_STEP_DATA \
$HI_RES_DIAGS \
$TIMESTEPS_FILE \
$NETCDF_OUTPUT \
--diags_config $DIAGS_CONFIG \
--n_sample_inits 48 \
--runner DataflowRunner \
--job_name one-step-diags-${USER}-${STAMP} \
--project vcm-ml \
--region us-central1 \
--temp_location gs://vcm-ml-scratch/tmp_dataflow \
--num_workers 4 \
--max_num_workers 30 \
--disk_size_gb 200 \
--worker_machine_type n2-highmem-4 \
--extra_package $(pwd)/external/report/dist/report-0.1.0.tar.gz \
--extra_package $(pwd)/external/gallery/dist/gallery-0.1.0.tar.gz \
--extra_package $(pwd)/workflows/one_step_diags/dist/one_step_diags-0.1.0.tar.gz
"

CMD="./dataflow.sh submit $PYTHON_CMD"
    
echo "Running command:"
echo ${CMD}

nohup $CMD &
