#!/bin/bash

set -e

# config file as argument

CONFIG=$1

# create the name of the experiment directory
EXP_PATH=$(python workflows/end_to_end/create_experiment_path.py --workflow-config $CONFIG)

echo $EXP_PATH

### Coarsening step
# get config
COARSEN_CONFIG=$(python workflows/end_to_end/get_step_inputs.py \
    --experiment-path ${EXP_PATH} \
    --workflow-config ${CONFIG} \
    --workflow-step "coarsen")

echo $COARSEN_CONFIG

read -ra ARR <<< "$COARSEN_CONFIG"

# run the step
./workflows/coarsen_restarts/submit_job.sh ${ARR[0]} ${ARR[1]} ${ARR[2]}
    
### One-step jobs
# get config

