#!/bin/bash

set -e

echo "Running and end-to-end pipeline:"
echo "Environment variables:"
echo "-------------------------------------------------------------------------------"
env
echo "-------------------------------------------------------------------------------"

TRAINING_TIMES="$(pwd)/training_times.json"
ONE_STEP_TIMES="$(pwd)/one_step_times.json"

echo "Generating timesteps:"
python3 workflows/end_to_end/generate_samples.py  \
    $CONFIG/time-control.yaml \
    > tmpconfig.json

jq .one_step tmpconfig.json > "$ONE_STEP_TIMES"
jq .train tmpconfig.json > "$TRAINING_TIMES"

echo "Using these timesteps for training"
cat "$TRAINING_TIMES"
echo ""
echo "Using these for one-step-jobs"
cat "$ONE_STEP_TIMES"
echo "-------------------------------------------------------------------------------"

echo "running the following end to end configuration:"
echo "-------------------------------------------------------------------------------"
export TRAINING_TIMES ONE_STEP_TIMES
envsubst < "workflows/end_to_end/end_to_end.yaml" | tee end-to-end.yml
echo "-------------------------------------------------------------------------------"

workflows/end_to_end/submit_workflow.sh end-to-end.yml