#!/bin/bash

if ! [[ -e data_old ]]
then
    mkdir data_old
    gsutil cp gs://vcm-ml-experiments/microphysics-emu-data/2021-07-29/validation_netcdfs/state_20160201.044500_0.nc data_old/
fi


cat << EOF > sample.yaml
use_wandb: false
batch_size: 128
epochs: 1
out_url: gs://vcm-ml-scratch/andrep/2021-10-02-wandb-training/dense
test_url: data_old
train_url: data_old
loss:
  optimizer:
    name: Adam
    kwargs:
      learning_rate: 0.0001
  loss_variables:
  - air_temperature_output
  - specific_humidity_output
  - cloud_water_mixing_ratio_output
  - total_precipitation
  weights:
    air_temperature_output: 500000.0
    specific_humidity_output: 500000.0
    cloud_water_mixing_ratio_output: 1.0
    total_precipitation: .04
  metric_variables:
  - tendency_of_air_temperature_due_to_microphysics
  - tendency_of_specific_humidity_due_to_microphysics
model:
  architecture:
    kwargs: {}
    name: dense
  input_variables:
  - air_temperature_input
  - specific_humidity_input
  - cloud_water_mixing_ratio_input
  - pressure_thickness_of_atmospheric_layer
  direct_out_variables:
  - cloud_water_mixing_ratio_output
  - total_precipitation
  residual_out_variables:
    air_temperature_output: air_temperature_input
    specific_humidity_output: specific_humidity_input
  tendency_outputs:
    air_temperature_output: tendency_of_air_temperature_due_to_microphysics
    specific_humidity_output: tendency_of_specific_humidity_due_to_microphysics
  normalize_key: mean_std
  enforce_positive: true
  selection_map:
    air_temperature_input: {stop: -10}
    cloud_water_mixing_ratio_input: {stop: -10}
    pressure_thickness_of_atmospheric_layer: {stop: -10}
    specific_humidity_input: {stop: -10}
  timestep_increment_sec: 900
transform:
  input_variables:
  - air_temperature_input
  - specific_humidity_input
  - cloud_water_mixing_ratio_input
  - pressure_thickness_of_atmospheric_layer
  output_variables:
  - cloud_water_mixing_ratio_output
  - total_precipitation
  - air_temperature_output
  - specific_humidity_output
  - tendency_of_air_temperature_due_to_microphysics
  - tendency_of_specific_humidity_due_to_microphysics
  antarctic_only: false
  derived_microphys_timestep: 900
  use_tensors: true
  vertical_subselections: null
EOF


python3 -m fv3fit.train_microphysics --config-path sample.yaml
