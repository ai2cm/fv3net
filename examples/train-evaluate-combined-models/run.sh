#!/bin/bash
set -e


EXPERIMENT=2021-02-15-multiple-model-template 
TRAIN_TEST_DATA=gs://vcm-ml-experiments/2020-12-17-n2f-moisture-conservation/control/nudge-to-fine-run

argo submit \
    --from workflowtemplate/train-diags-prog-multiple-models \
    -p root=gs://vcm-ml-scratch/annak/$EXPERIMENT \
    -p train-test-data=$TRAIN_TEST_DATA \
    -p training-configs="$( yq . config-list.yml )" \
    -p reference-restarts=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
    -p initial-condition="20160805.000000" \
    -p prognostic-run-config="$(< prognostic-run-short.yaml)" \
    -p training-flags="--local-download-path train-data-download-dir" \
    -p train-times="$(< train_short.json)" \
    -p test-times="$(< test_short.json)" \
    -p public-report-output=gs://vcm-ml-public/annak/$EXPERIMENT  \
    -p segment-count=1 \
    -p memory-training="10Gi" \
    -p memory-offline-diags="10Gi" \
    --name test-multiple-model-template
