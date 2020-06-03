#!/bin/bash

set -x
set -e

TRAINING_DATA=gs://vcm-ml-data/testing-noah/7563d34ad7dd6bcc716202a0f0c8123653f50ca4/training_data/
OUTPUT=gs://vcm-ml-scratch/annak/2020-05-22/sklearn_train/

#mkdir -p data
#gsutil -m rsync -d -r "$TRAINING_DATA" data

python -m fv3net.regression.sklearn \
    data \
    ../../workflows/end_to_end/kustomization/train_sklearn_model.yml  \
    $OUTPUT