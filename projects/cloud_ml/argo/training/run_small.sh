#!/bin/bash

set -e

TRAINING=$1

BUCKET="vcm-ml-experiments"
PROJECT="cloud-ml"
EXPERIMENT="fine-cloud-${TRAINING}-local"
TRIAL="trial-0"
TAG="${EXPERIMENT}"  # required
NAME=train-cloud-ml-${TRAINING}-$(openssl rand --hex 2) # required
WANDB_PROJECT="radiation-cloud-ml"

if [[ "$TRAINING" == *"rf"* ]]; then
    VALIDATION_ARG=" "
else
    VALIDATION_ARG="$( yq . validation_small.yaml )"
fi

argo submit --from workflowtemplate/training \
    -p training_config="$( yq . ${TRAINING}-training-config.yaml )" \
    -p training_data_config="$( yq . train_small.yaml )" \
    -p validation_data_config="${VALIDATION_ARG}" \
    -p output="gs://${BUCKET}/${PROJECT}/$(date +%F)/${TAG}/trained_models/${TRAINING}" \
    -p memory="25Gi" \
    -p cpu="4000m" \
    -p flags="--cache.local_download_path train-data-download-dir" \
    -p wandb-project="${WANDB_PROJECT}" \
    --name "${NAME}" \
    --labels "project=${PROJECT},experiment=${EXPERIMENT},trial=${TRIAL}"