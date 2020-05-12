#!/bin/bash

set -x
set -e

TRAINED_MODEL=gs://vcm-ml-data/test-end-to-end-integration/physics-off/train_sklearn_model
TEST_DATA_PATH=gs://vcm-ml-scratch/annak/temp_data
HIGH_RES_DATA_PATH=gs://vcm-ml-data/orchestration-testing/shield-coarsened-diags-2019-12-04
VARIABLE_NAMES_FILE=workflows/end_to_end/kustomization/test_sklearn_variable_names.yml
IMAGE=us.gcr.io/vcm-ml/fv3net:6f59138183398745b84ab582d6fcaebe208f5d6a
OUTPUT=gs://vcm-ml-scratch/annak/2020-05-12/argo_test

argo submit argo.yaml \
    -p trained_model=$TRAINED_MODEL \
    -p testing_data=$TEST_DATA_PATH \
    -p diagnostics_data=$HIGH_RES_DATA_PATH \
    -p output=$OUTPUT \
    -p image=$IMAGE
