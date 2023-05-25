#!/bin/bash

set -e

CONFIG=$1

PROJECT=cloud-ml

EXPERIMENT="nudged-radiation-port"
TRIAL="trial-0"
TAG=${EXPERIMENT}-${CONFIG}
NAME="${TAG}-$(openssl rand --hex 2)"

argo submit --from workflowtemplate/prognostic-run \
    -p project=${PROJECT} \
    -p tag=${TAG} \
    -p config="$(< ${CONFIG}-config.yaml)" \
    -p segment-count="39" \
    -p memory="20Gi" \
    --name "${NAME}" \
    --labels "project=${PROJECT},experiment=${EXPERIMENT},trial=${TRIAL}"