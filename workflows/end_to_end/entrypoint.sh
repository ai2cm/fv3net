#!/bin/bash

set -e

echo "Running an end-to-end pipeline:"
echo "Environment variables:"
echo "-------------------------------------------------------------------------------"
env
echo "-------------------------------------------------------------------------------"

TRAIN_AND_TEST_TIMES="$(pwd)/train_and_test_times.json"
ONE_STEP_TIMES="$(pwd)/one_step_times.json"

echo "Generating timesteps:"
python3 workflows/end_to_end/generate_samples.py  \
    $CONFIG/time-control.yaml \
    > tmpconfig.json

jq .one_step tmpconfig.json > "$ONE_STEP_TIMES"
jq .train_and_test tmpconfig.json > "$TRAIN_AND_TEST_TIMES"

echo "Using these timesteps for training and testing"
cat "$TRAIN_AND_TEST_TIMES"
echo ""
echo "Using these for one-step-jobs"
cat "$ONE_STEP_TIMES"
echo "-------------------------------------------------------------------------------"

echo "running the following end to end configuration:"
echo "-------------------------------------------------------------------------------"
export TRAIN_AND_TEST_TIMES ONE_STEP_TIMES
envsubst < "$CONFIG/end_to_end.yaml" | tee end-to-end.yml
echo "-------------------------------------------------------------------------------"

workflows/end_to_end/submit_workflow.sh end-to-end.yml
