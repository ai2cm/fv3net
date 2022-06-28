#!/bin/bash

set -e

CONFIG=$1

PROJECT=cloud-ml

EXPERIMENT="cloud-ml-training-data"
TRIAL="trial-0"
TAG=${EXPERIMENT}
NAME="${TAG}-nudge-to-fine-run"

argo submit --from workflowtemplate/prognostic-run \
    -p project=${PROJECT} \
    -p tag=${TAG} \
    -p config="$(< ${CONFIG}-config.yaml)" \
    -p segment-count="2" \
    --name "${NAME}" \
    --labels "project=${PROJECT},experiment=${EXPERIMENT},trial=${TRIAL}"