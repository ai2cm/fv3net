#!/bin/bash

set -e

echo "Running an nudging integration pipeline:"
echo "Environment variables:"
echo "-------------------------------------------------------------------------------"
env
echo "-------------------------------------------------------------------------------"
echo "\n\n"

echo "running the following end to end configuration:"
echo "-------------------------------------------------------------------------------"
envsubst < "$CONFIG/end_to_end.yaml" | tee end-to-end.yml
echo "-------------------------------------------------------------------------------"

workflows/end_to_end/submit_workflow.sh end-to-end.yml
