#!/bin/bash

set -e

EXPERIMENT=2020-09-25-physics-on-free

argo submit \
    --from workflowtemplate/train-diags-prog \
    -p image-tag=dbf8db14a939ea59f460345e65cbe46c03c2c1f6 \
    -p root=gs://vcm-ml-experiments/$EXPERIMENT \
    -p train-test-data=gs://vcm-ml-archive/prognostic_runs/2020-09-25-physics-on-free/ \
    -p training-config="$(< training-config.yaml)" \
    -p reference-restarts=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
    -p initial-condition="20160805.000000" \
    -p prognostic-run-config="$(< prognostic-run.yaml)" \
    -p train-times="$(<  train.json)" \
    -p test-times="$(<  test.json)" \
    -p public-report-output=gs://vcm-ml-public/offline_ml_diags/$EXPERIMENT \
    -p segment-count=1

