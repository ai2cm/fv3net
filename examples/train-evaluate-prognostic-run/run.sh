#!/bin/bash

set -e

EXPERIMENT=2020-09-25-physics-on-free-debug

argo submit \
    --from workflowtemplate/train-diags-prog \
    -p image-tag=4cbe9d6f9df349afa8e376e094be5d1ff6a0c7c3 \
    -p root=gs://vcm-ml-scratch/noah/$EXPERIMENT \
    -p train-test-data=gs://vcm-ml-archive/prognostic_runs/2020-09-25-physics-on-free/ \
    -p training-config="$(< training-config.yaml)" \
    -p reference-restarts=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
    -p initial-condition="20160805.000000" \
    -p prognostic-run-config="$(< prognostic-run.yaml)" \
    -p train-times="$(<  train.json)" \
    -p test-times="$(<  test.json)" \
    -p public-report-output=gs://vcm-ml-scratch/noah/offline_ml_diags/$EXPERIMENT \
    -p segment-count=1

