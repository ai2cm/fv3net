#!/bin/bash

MODEL=gs://vcm-ml-experiments/2020-06-17-triad-round-1/nudging/train_sklearn_model/sklearn_model.pkl
OUTPUT=gs://vcm-ml-scratch/annak/2020-07-08/test-new-features-nudging/control
gsutil -m rm -r $OUTPUT
python -m offline_ml_diags.compute_diags \
    workflows/offline_ml_diags/tests/test_nudging_config.yml \
    $MODEL \
    $OUTPUT \
    --timesteps-file workflows/offline_ml_diags/tests/times_nudging.json
