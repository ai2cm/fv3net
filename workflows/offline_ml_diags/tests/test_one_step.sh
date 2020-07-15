#!/bin/bash

MODEL=gs://vcm-ml-scratch/annak/sklearn_train/one_step/sklearn_model.pkl
OUTPUT=gs://vcm-ml-scratch/annak/test-offline-validation-workflow/one_step/
python -m offline_ml_diags.compute_diags \
    workflows/offline_ml_diags/tests/test_one_step_config.yml \
    $MODEL \
    $OUTPUT \
   --timesteps-file workflows/offline_ml_diags/tests/times_one_step.json
