#!/bin/bash

set -e

EXPERIMENT=2020-10-30-nudge-to-obs-GRL-paper/rf-default

argo submit \
    --from workflowtemplate/train-diags-prog \
    -p root=gs://vcm-ml-experiments/$EXPERIMENT \
    -p train-test-data=gs://vcm-ml-experiments/2020-10-30-nudge-to-obs-GRL-paper/nudge-to-obs-run \
    -p training-config="$(< training-config.yaml)" \
    -p reference-restarts=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
    -p initial-condition="20160101.000000" \
    -p prognostic-run-config="$(< prognostic-run.yaml)" \
    -p train-times="$(<  ../../train.json)" \
    -p test-times="$(<  ../../test.json)" \
    -p public-report-output=gs://vcm-ml-public/offline_ml_diags/$EXPERIMENT \
    -p segment-count=12 \
    -p cpu-prog=6 \
    -p memory-prog="15Gi" \
    -p flags="--nudge-to-observations" \
    -p chunks="$(< chunks.yaml)"

