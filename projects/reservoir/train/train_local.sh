#!/bin/bash

CURRENT_DATE=$(date +%Y%m%d)
export WANDB_PROJECT=sst-reservoir-training
export WANDB_ENTITY=ai2cm
export WANDB_RUN_GROUP=$CURRENT_DATE-v3
export WANDB_MODE=online

EXPERIMENT="sst-reservoir-training"
NAME="pure-8x8sub-halo6-state1000"
RANDOM_TAG=$(openssl rand -hex 3)
OUTPUT_URL="gs://vcm-ml-experiments/${EXPERIMENT}/${CURRENT_DATE}/${NAME}"

train_config=training-config.yaml
training_data=training-data.yaml
validation_data=validation-data.yaml

config_dir=$1

# Check if the argument was provided
if [ -z "$config_dir" ]; then
  echo "Please provide the configuration yaml directory as an argument."
  exit 1
fi

# Loop through each tile and submit the specified number of jobs
for tile in {0..5}; do
  # Create a temporary directory for the updated configuration files
  export TILE=$tile

  tmpdir=$(mktemp -d)
  envsubst < $config_dir/$training_data > $tmpdir/$training_data
  envsubst < $config_dir/$validation_data > $tmpdir/$validation_data
  envsubst < $config_dir/$train_config > $tmpdir/$train_config

  export WANDB_NAME="${NAME}-tile${tile}-${RANDOM_TAG}"
  export SUBMIT_DIR=$(pwd)

  (
    cd $tmpdir && \
    python3 -m fv3fit.train \
        $train_config \
        $training_data \
        "$OUTPUT_URL-tile${TILE}" \
        --validation-data-config $validation_data > ${SUBMIT_DIR}/log.tile${tile}.txt 2>&1 &
  )

done

