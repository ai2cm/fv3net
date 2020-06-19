#!/bin/bash

MODEL=gs://vcm-ml-scratch/noah/2020-06-17-triad-round1/trained/sklearn_model.pkl
OUTPUT=gs://vcm-ml-scratch/annak/test-offline-validation-workflow/fineres/

python -m offline_ml_diags.compute_diags \
    workflows/offline_ml_diags/tests/test_fine_res_config.yml \
    $MODEL \
    $OUTPUT \
   --timesteps-file workflows/offline_ml_diags/tests/times_fine_res.json
