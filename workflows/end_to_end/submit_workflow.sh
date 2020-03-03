#!/bin/bash

export PYTHONPATH="./"

set -e


config_file=$1

# get arguments for all steps
ALL=$(python workflows/end_to_end/get_experiment_steps_and_args.py ${config_file})
NAME=$(echo ${ALL} | jq -r .name)
WORKFLOW=$(echo ${ALL} | jq -r .workflow)
ALL_CMD_AND_ARGS=$(echo ${ALL} | jq -r .command_and_args)

LOGDIR="./logs"
if [ ! -d $LOGDIR ]; then
  mkdir $LOGDIR
fi
LOGFILE="$LOGDIR/${NAME}.log"
exec > >(tee ${LOGFILE}) 2>&1

# run the workflow
echo -e "\n\n\n############### Starting workflow "${NAME}" to execute the following steps: "${WORKFLOW} "\n\n\n"


echo -e "\n\n\n############### Creating fv3net python package sdist files...\n\n\n"

# extra fv3net packages
(
  cd external/vcm
  python setup.py sdist
)
(
  cd external/vcm/external/mappm
  python setup.py sdist
)

for STEP in ${WORKFLOW}; do
  STEP_COMMAND_ARGS=$(echo ${ALL_CMD_AND_ARGS} | jq -r .${STEP})
  echo -e "\n\n\n############### Starting step "${STEP}"... \n\n\n"
  echo -e "############### Running command \"${STEP_COMMAND_ARGS}\"\n\n\n"
  $STEP_COMMAND_ARGS
  echo -e "\n\n\n"
done

echo "############### ...workflow completed."