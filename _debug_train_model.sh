#!/bin/bash

# Put config in a HEREDOC
cat <<EOF >config.yaml
model_type: sklearn_random_forest
hyperparameters:
  max_depth: 4
  n_estimators: 2
input_variables:
- air_temperature
- specific_humidity
output_variables:
- dQ1
- dQ2
batch_function: batches_from_mapper
batch_kwargs:
  num_batches: 1
  timesteps_per_batch: 1
  init_time_dim_name: "initial_time"
  mapping_function: open_fine_res_apparent_sources
EOF

FINE_RES=gs://vcm-ml-experiments/2020-06-02-fine-res/fine_res_budget/
TRAINED_ML=gs://vcm-ml-experiments/2020-06-02-fine-res/trained_sklearn/

python -m fv3net.regression.sklearn \
    "$FINE_RES" config.yaml "$TRAINED_ML" \
    --no-train-subdir-append
