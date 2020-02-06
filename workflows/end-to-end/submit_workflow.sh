#!/bin/bash

set -e

# config file as argument

CONFIG=$1

# create the name of the experiment directory
EXP_PATH=$(python workflows/end-to-end/create_experiment_path.py --workflow-config $CONFIG)

echo $EXP_PATH

### Coarsening step
# get config
COARSEN_CONFIG=$(python workflows/end-to-end/get_step_inputs.py \
    --experiment-path ${EXP_PATH} \
    --workflow-config ${CONFIG} \
    --workflow-step "coarsen")

echo ${COARSEN_CONFIG}

# run the step
#workflows/coarsen_restarts/submit_job.sh $COARSEN_INPUT $COARSEN_OUTPUT $COARSEN_METHOD 
    
### One-step jobs
# get config

