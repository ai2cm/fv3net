#!/bin/bash

set -e

config_file=$1

# get arguments for all steps
output_root=$(python workflows/end_to_end/create_experiment_path.py $config_file)

echo $output_root
coarsen_args=$(echo $output_root | jq -r .coarsen_restarts)
workflows/coarsen_restarts/orchestrator_job.sh $coarsen_args
