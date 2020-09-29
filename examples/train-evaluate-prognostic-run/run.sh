#!/bin/bash

set -e
set -x

EXPERIMENT=2020-09-29-physics-on-free-moreInputs-moreData

argo submit \
    --from workflowtemplate/train-diags-prog \
    -p image-tag=1e3ba79515f1dce4072c57e3af14455e63807500 \
    -p root=gs://vcm-ml-experiments/noah/$EXPERIMENT \
    -p train-test-data=gs://vcm-ml-archive/prognostic_runs/2020-09-25-physics-on-free/ \
    -p training-config="$(< training-config.yaml)" \
    -p reference-restarts=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
    -p initial-condition="20160805.000000" \
    -p prognostic-run-config="$(< prognostic-run.yaml)" \
    -p train-times="$(<  train.json)" \
    -p test-times="$(<  test.json)" \
    -p public-report-output=gs://vcm-ml-public/offline_ml_diags/$EXPERIMENT \
    -p segment-count=1

