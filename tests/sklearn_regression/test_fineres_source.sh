#!/bin/bash

TRAINING_DATA=gs://vcm-ml-scratch/noah/2020-05-19
OUTPUT=gs://vcm-ml-scratch/annak/2020-05-22/sklearn_train/

python -m fv3net.regression.sklearn \
    $TRAINING_DATA \
    train_sklearn_model_fineres_source.yml  \
    $OUTPUT \
    --no-train-subdir-append