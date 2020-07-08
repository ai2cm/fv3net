#!/bin/bash

MODEL=gs://vcm-ml-experiments/2020-06-17-triad-round-1/one-step-clouds-off/train_sklearn_model/sklearn_model.pkl
OUTPUT=gs://vcm-ml-scratch/annak/test-offline-validation-workflow/one_step/
python -m offline_ml_diags.compute_diags \
    workflows/offline_ml_diags/tests/test_one_step_config.yml \
    $MODEL \
    $OUTPUT \
   --timesteps-file workflows/offline_ml_diags/tests/times_one_step.json
