#!/bin/bash

set -x
set -e

TRAINING_DATA=gs://vcm-ml-data/testing-noah/2020-04-19/07c346d0d59183c0216591e0e49d7cd19edd0976/training_data/
OUTPUT=gs://vcm-ml-data/testing-noah/$(git rev-parse HEAD)/sklearn_train/

mkdir -p data
gsutil -m rsync -r "$TRAINING_DATA" data

python -m fv3net.regression.sklearn.train \
    data \
    train_sklearn_model.yml \
    $OUTPUT
