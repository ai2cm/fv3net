#!/bin/bash

set -e

BUCKET=vcm-ml-scratch # don't pass bucket arg to use default 'vcm-ml-experiments'
PROJECT=brianh # don't pass project arg to use default 'default'
TAG=fine-res-test-train-diags-prog # required
NAME="${TAG}-train-diags-prog-$(openssl rand --hex 6)"

argo submit --from workflowtemplate/train-diags-prog \
    -p bucket=${BUCKET} \
    -p project=${PROJECT} \
    -p tag=${TAG} \
    -p train-test-data=gs://vcm-ml-experiments/2020-07-30-fine-res \
    -p training-configs="$( yq . training-config.yaml )" \
    -p training-flags="--local-download-path train-data-download-dir" \
    -p reference-restarts=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
    -p initial-condition="20160805.000000" \
    -p prognostic-run-config="$(< prognostic-run.yaml)" \
    -p train-times="$(<  ../../train_short.json)" \
    -p test-times="$(<  ../../test_short.json)" \
    -p public-report-output=gs://vcm-ml-public/offline_ml_diags/$TAG \
    -p segment-count=1 \
    --name "${NAME}"
