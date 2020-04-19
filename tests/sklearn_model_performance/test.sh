#!/bin/bash

set -x
set -e

TRAINED_MODEL=gs://vcm-ml-data/testing-noah/075f154fb633469e56be48faedc196756e67179a/sklearn_train/
TRAINING_DATA=gs://vcm-ml-data/testing-noah/7563d34ad7dd6bcc716202a0f0c8123653f50ca4/training_data/
DIAG_DATA=gs://vcm-ml-data/testing-noah/ceb320ffa48b8f8507b351387ffa47d1b05cd402/coarsen_diagnostics/gfsphysics_15min_coarse.zarr

OUTPUT=gs://vcm-ml-data/testing-noah/$(git rev-parse HEAD)/sklearn_model_performance

# mkdir -p data
# gsutil -m rsync -d -r "$TRAINING_DATA" data


python -m fv3net.diagnostics.sklearn_model_performance \
    --num_test_zarrs 1 \
    $TRAINED_MODEL \
    $TRAINING_DATA \
    $DIAG_DATA \
    test_sklearn_variable_names.yml \
    $OUTPUT
