#!/bin/bash

MODEL=gs://vcm-ml-scratch/andrep/test-nudging-workflow/train_sklearn_model/sklearn_model.pkl
OUTPUT=gs://vcm-ml-scratch/annak/test-offline-validation-workflow/nudging/

gsutil -m rm -r $OUTPUT
python -m offline_ml_diags.compute_diags \
    workflows/offline_ml_diags/tests/test_nudging_config.yml \
    $MODEL \
    $OUTPUT 
