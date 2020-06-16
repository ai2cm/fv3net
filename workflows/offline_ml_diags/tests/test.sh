#!/bin/bash

TEST_DATA=gs://vcm-ml-scratch/andrep/test-nudging-workflow/nudging
MODEL=gs://vcm-ml-scratch/andrep/test-nudging-workflow/train_sklearn_model/sklearn_model.pkl
OUTPUT=gs://vcm-ml-scratch/annak/test-offline-validation-workflow

python ../../workflows/offline_validation_diags/compute_diags.py \
    config.yml \
    $MODEL \
    $OUTPUT \
    --timesteps-file times.json