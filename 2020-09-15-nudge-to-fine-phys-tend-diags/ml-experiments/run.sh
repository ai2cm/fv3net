#!/bin/bash

set -e

FV3NET_HASH=b8816a17832746b7cb4185a56e22c9baec5581bb
INITIAL_CONDITION=20160805.000000
REFERENCE_RESTARTS=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts

TRAINING_CONFIG="$(< $1)"
PROGNOSTIC_RUN_CONFIG="$(< $2)"
TRAINING_TIMES="$(< $3)"
TESTING_TIMES="$(< $4)"
EXPERIMENT=$5

argo submit --from workflowtemplate/train-diags-prog \
    -p image-tag=$FV3NET_HASH \
    -p root=gs://vcm-ml-experiments/2020-09-16-$EXPERIMENT \
    -p train-test-data=notused \
    -p store-true-args=" " \
    -p training-config="$TRAINING_CONFIG" \
    -p train-times="$TRAINING_TIMES" \
    -p initial-condition=$INITIAL_CONDITION \
    -p prognostic-run-config="$PROGNOSTIC_RUN_CONFIG" \
    -p reference-restarts=$REFERENCE_RESTARTS \
    -p test-times="$TESTING_TIMES" \
    -p public-report-output=gs://vcm-ml-public/spencerc/2020-09-16-$EXPERIMENT
