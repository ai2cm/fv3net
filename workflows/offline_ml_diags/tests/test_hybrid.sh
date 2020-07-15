#!/bin/bash

MODEL=gs://vcm-ml-scratch/noah/dev/2020-07-09-saved-fine-sres-baseline-data-trained-sklearn/sklearn_model.pkl
OUTPUT=gs://vcm-ml-scratch/brianh/offline-ML-diags-testing/
gsutil -m rm -r $OUTPUT
python -m offline_ml_diags.compute_diags \
    workflows/offline_ml_diags/tests/test_hybrid_config.yml \
    $MODEL \
    $OUTPUT
