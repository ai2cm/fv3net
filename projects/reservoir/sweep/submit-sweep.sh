#!/bin/bash

# Define the number of jobs to submit for each tile
num_jobs=6

sweep_config=tile-train-sweep.yaml
training_data=training-data.yaml
validation_data=validation-data.yaml

temp_config=temporary_config_files

# Loop through each tile and submit the specified number of jobs
for tile in {0..5}; do
  # Create a temporary directory for the updated configuration files
  mkdir -p $temp_config
  python format_for_tile.py $sweep_config $tile name > $temp_config/$sweep_config
  python format_for_tile.py $training_data $tile filepath > $temp_config/$training_data
  python format_for_tile.py $validation_data $tile filepath > $temp_config/$validation_data

  cd $temp_config
  wandb sweep $sweep_config &> sweep.log
  sweep_id=$(tail -n 1 sweep.log | grep -oP '(?<=wandb agent ).*')
  echo $sweep_id
  cd ..

  # Submit the specified number of jobs for the current tile using the updated configuration files
  for ((i=1; i<=num_jobs; i++)); do
    argo submit argo.yaml \
      -p sweep-id="$sweep_id" \
      -p sweep-config="$(cat $temp_config/$sweep_config)" \
      -p training-config="$(cat training-config.yaml)" \
      -p training-data-config="$(cat $temp_config/$training_data)" \
      -p validation-data-config="$(cat $temp_config/$validation_data)" > /dev/null
    echo "Submitting job $i for tile $tile"
  done
done

