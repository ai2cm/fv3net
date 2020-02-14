#!/bin/bash

set -e

config_file=$1

# get arguments for all steps
ALL=$(python workflows/end_to_end/get_experiment_steps_and_args.py $config_file)
WORKFLOW=$(echo $ALL | jq -r .workflow)
COMMANDS=$(echo $ALL | jq -r .commands)
ALL_ARGUMENTS=$(echo $ALL | jq -r .arguments)

# extra fv3net packages
cd external/vcm
python setup.py sdist

cd external/mappm
python setup.py sdist

cd ../../../..

echo "Starting workflow to execute the following steps: "$WORKFLOW
for STEP in $WORKFLOW; do
  echo "Starting command "$STEP"..."
  COMMAND=$(echo $COMMANDS | jq -r .${STEP})
  ARGUMENTS=$(echo $ALL_ARGUMENTS | jq -r .${STEP})
  echo "Running command \""${COMMAND}" "${ARGUMENTS}"\""
  $($COMMAND $ARGUMENTS)
done
