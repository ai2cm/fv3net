#!/bin/bash

set -e

config_file=$1

# get arguments for all steps
all_step_args=$(python workflows/end_to_end/create_experiment_path.py $config_file)
echo $all_step_args

# Coarsening Step
coarsen_args=$(echo $all_step_args | jq -r .coarsen_restarts)
workflows/coarsen_restarts/orchestrator_job.sh $coarsen_args

# One-step Jobs
one_step_args=$(echo $all_step_args | jq -r .one_step_run)


