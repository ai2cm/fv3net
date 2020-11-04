#!/bin/bash

set -e

EXPERIMENT=2020-test

argo submit \
    --from workflowtemplate/train-diags-prog \
    -p root=gs://vcm-ml-scratch/$EXPERIMENT \
    -p train-test-data=gs://vcm-ml-experiments/2020-07-30-fine-res \
    -p training-config="$(< training-config.yaml)" \
    -p reference-restarts=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
    -p initial-condition="20160805.000000" \
    -p prognostic-run-config="$(< prognostic-run.yaml)" \
    -p train-times="$(<  ../../train_short.json)" \
    -p test-times="$(<  ../../test_short.json)" \
    -p public-report-output=gs://vcm-ml-public/offline_ml_diags/$EXPERIMENT \
    -p segment-count=1

