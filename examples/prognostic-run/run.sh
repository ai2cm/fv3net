#!/bin/bash

set -e

# adjust experiment name and replace "vcm-ml-scratch" with "vcm-ml-experiments"
# under output after debugging configuration
EXPERIMENT=fill_in_here

argo submit \
    --from workflowtemplate/prognostic-run \
    -p image-tag=4a8fc6aa3337d51967d5483695be582dd09561db \
    -p output=gs://vcm-ml-scratch/$EXPERIMENT \
    -p trained-ml=gs://vcm-ml-scratch/test-end-to-end-integration/integration-test-ffca74392a57/trained_model \
    -p reference-restarts=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
    -p initial-condition="20160805.000000" \
    -p config="$(< prognostic-run.yaml)" \
    -p segment-count=1
