#!/bin/bash

set -e

CONFIG=$1

PROJECT=cloud-ml

EXPERIMENT="nudge-to-fine-radiation-port"
TRIAL="trial-0"
TAG=${EXPERIMENT}-${CONFIG}
NAME="${TAG}-$(openssl rand --hex 2)"

argo submit --from workflowtemplate/prognostic-run \
    -p project=${PROJECT} \
    -p tag=${TAG} \
    -p config="$(< ${CONFIG}-config.yaml)" \
    -p segment-count="4" \
    -p memory="15Gi" \
    --name "${NAME}" \
    --labels "project=${PROJECT},experiment=${EXPERIMENT},trial=${TRIAL}"