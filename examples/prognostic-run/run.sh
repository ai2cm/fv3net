#!/bin/bash

set -e

MODEL_URL=gs://vcm-ml-scratch/test-end-to-end-integration/integration-test-2c1ba84b1350/trained_model

argo submit \
    --from workflowtemplate/prognostic-run \
    -p output=gs://vcm-ml-scratch/test-prognostic-run-example \
    -p reference-restarts=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
    -p initial-condition="20160805.000000" \
    -p config="$(< prognostic-run.yaml)" \
    -p segment-count=1 \
    -p flags="--model_url ${MODEL_URL}"
