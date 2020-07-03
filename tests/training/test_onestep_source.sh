#!/bin/bash

set -x
set -e

TRAINING_DATA=gs://vcm-ml-data/testing-noah/7563d34ad7dd6bcc716202a0f0c8123653f50ca4/training_data/
OUTPUT=gs://vcm-ml-scratch/annak/sklearn_train/one_step

gsutil -m rm -r $OUTPUT
python -m fv3net.regression.sklearn \
    $TRAINING_DATA \
    tests/sklearn_regression/train_sklearn_model_onestep_source.yml  \
    $OUTPUT