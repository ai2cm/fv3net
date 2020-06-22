#!/bin/bash

TRAINING_DATA=gs://vcm-ml-experiments/2020-06-02-fine-res/fine_res_budget/
OUTPUT=gs://vcm-ml-scratch/annak/2020-05-22/sklearn_train/

python -m fv3net.regression.sklearn \
    $TRAINING_DATA \
    tests/sklearn_regression/train_sklearn_model_fineres_source.yml  \
    $OUTPUT \
    --no-train-subdir-append \
    --timesteps-file times.json