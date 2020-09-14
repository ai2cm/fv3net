#!/bin/bash

set -e

EXPERIMENT=2020-09-03-ignore-physics

cat << EOF > training-config.yaml
model_type: sklearn_random_forest
hyperparameters:
  max_depth: 13
  n_estimators: 1
input_variables:
- air_temperature
- specific_humidity
output_variables:
- dQ1
- dQ2
batch_function: batches_from_geodata
batch_kwargs:
  timesteps_per_batch: 10
  mapping_function: open_fine_res_apparent_sources
  mapping_kwargs:
    offset_seconds: 450
    rename_vars:
      delp: pressure_thickness_of_atmospheric_layer
EOF

cat << EOF > prognostic-run.yaml
base_version: v0.5
namelist:
  coupler_nml:
    days: 1
    hours: 0
    minutes: 0
    seconds: 0
  diag_manager_nml:
    flush_nc_files: true
  fv_core_nml:
    do_sat_adj: false
  gfdl_cloud_microphysics_nml:
    fast_sat_adj: false
EOF


# https://github.com/VulcanClimateModeling/fv3net/pull/615

argo submit \
    --from workflowtemplate/train-diags-prog \
    -p image-tag=bdafdfb3ee3c4a7e15849b01aefc28185c69defc \
    -p root=gs://vcm-ml-experiments/$EXPERIMENT \
    -p train-test-data=gs://vcm-ml-experiments/2020-07-30-fine-res \
    -p training-config="$(< training-config.yaml)" \
    -p reference-restarts=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
    -p initial-condition="20160805.000000" \
    -p prognostic-run-config="$(< prognostic-run.yaml)" \
    -p train-times="$(<  ../train_short.json)" \
    -p test-times="$(<  ../test_short.json)" \
    -p public-report-output=gs://vcm-ml-public/offline_ml_diags/$EXPERIMENT

