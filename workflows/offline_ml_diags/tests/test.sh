#!/bin/bash

#MODEL=gs://vcm-ml-scratch/andrep/test-nudging-workflow/train_sklearn_model/sklearn_model.pkl
MODEL=gs://vcm-ml-scratch/noah/2020-06-17-triad-round1/trained
OUTPUT=gs://vcm-ml-scratch/annak/test-offline-validation-workflow/fineres

python -m offline_ml_diags.compute_diags \
    workflows/offline_ml_diags/tests/fine_res_config.yml \
    $MODEL \
    $OUTPUT \
    --timesteps-file workflows/offline_ml_diags/tests/times.json
