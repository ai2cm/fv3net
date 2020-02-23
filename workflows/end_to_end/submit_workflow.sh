#!/bin/bash

export PYTHONPATH="./"

set -e


config_file=$1

# get arguments for all steps
ALL=$(python workflows/end_to_end/get_experiment_steps_and_args.py ${config_file})
NAME=$(echo ${ALL} | jq -r .name)
WORKFLOW=$(echo ${ALL} | jq -r .workflow)
COMMANDS=$(echo ${ALL} | jq -r .commands)
ALL_ARGUMENTS=$(echo ${ALL} | jq -r .arguments)

LOGFILE=${NAME}".log"
exec > >(tee ${LOGFILE}) 2>&1

# run the workflow
echo -e "\n\n\n############### Starting workflow "${NAME}" to execute the following steps: "${WORKFLOW} "\n\n\n"


echo -e "\n\n\n############### Creating fv3net python package sdist files...\n\n\n"

# extra fv3net packages
(
  cd external/vcm
  python setup.py sdist

  cd external/mappm
  python setup.py sdist
)

for STEP in ${WORKFLOW}; do
  echo -e "\n\n\n############### Starting step "${STEP}"... \n\n\n"
  COMMAND=$(echo ${COMMANDS} | jq -r .${STEP})
  ARGUMENTS=$(echo ${ALL_ARGUMENTS} | jq -r .${STEP})
  echo -e "############### Running command \""${COMMAND}" "${ARGUMENTS}"\"\n\n\n"
  ${COMMAND} ${ARGUMENTS}
  echo -e "\n\n\n"
done

echo "############### ...workflow completed."