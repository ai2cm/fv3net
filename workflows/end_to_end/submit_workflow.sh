#!/bin/bash

export PYTHONPATH="./"

set -e

config_file=$1

# get arguments for all steps
ALL=$(python workflows/end_to_end/get_experiment_steps_and_args.py $config_file)
WORKFLOW=$(echo $ALL | jq -r .workflow)
COMMANDS=$(echo $ALL | jq -r .commands)
ALL_ARGUMENTS=$(echo $ALL | jq -r .arguments)

# run the workflow
echo -e "\n\n\n"
echo "Starting workflow to execute the following steps: "$WORKFLOW
echo -e "\n\n\n"

# extra fv3net packages
cd external/vcm
python setup.py sdist

cd external/mappm
python setup.py sdist

cd ../../../..

for STEP in $WORKFLOW; do
  echo -e "\n\n\n"
  echo "Starting step "$STEP"..."
  echo -e "\n\n\n"
  COMMAND=$(echo $COMMANDS | jq -r .${STEP})
  ARGUMENTS=$(echo $ALL_ARGUMENTS | jq -r .${STEP})
  echo "Running command \""${COMMAND}" "${ARGUMENTS}"\""
  echo -e "\n\n\n"
  $COMMAND $ARGUMENTS
  echo -e "\n\n\n"
done

echo "...workflow completed."