#!/bin/bash

set -e

BUCKET=vcm-ml-scratch # don't pass bucket arg to use default 'vcm-ml-experiments'
PROJECT=brianh # don't pass project arg to use default 'default'
TAG=n2f-3km-test-prog-run # required
MODEL_URL=gs://vcm-ml-scratch/test-end-to-end-integration/2021-04-28/e387b3b85c9b33fdc5a4196418094e94f6829aef/trained_models/keras_dQ2_model
NAME="${TAG}-prognostic-run-$(openssl rand --hex 6)"

argo submit --from workflowtemplate/prognostic-run \
    -p bucket=${BUCKET} \
    -p project=${PROJECT} \
    -p tag=${TAG} \
    -p reference-restarts=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
    -p initial-condition="20160805.000000" \
    -p config="$(< prognostic-run.yaml)" \
    -p segment-count=1 \
    -p flags="--model_url ${MODEL_URL}" \
    --name "${NAME}"
