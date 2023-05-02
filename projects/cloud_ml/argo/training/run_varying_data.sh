#!/bin/bash

set -e

TRAINING=$1
DATA=$2

BUCKET="vcm-ml-experiments"
PROJECT="cloud-ml"
EXPERIMENT="fine-cloud-${TRAINING}-local-${DATA}"
TRIAL="trial-0"
TAG="${EXPERIMENT}"  # required
NAME=train-cloud-ml-${TRAINING}-local-${DATA}-$(openssl rand --hex 2) # required
WANDB_PROJECT="radiation-cloud-ml"

if [[ "$TRAINING" == *"rf"* ]]; then
    VALIDATION_ARG=" "
else
    VALIDATION_ARG="$( yq . validation.yaml )"
fi

argo submit --from workflowtemplate/training \
    -p training_config="$( yq . ${TRAINING}-training-config.yaml )" \
    -p training_data_config="$( yq . train_${DATA}.yaml )" \
    -p validation_data_config="${VALIDATION_ARG}" \
    -p output="gs://${BUCKET}/${PROJECT}/$(date +%F)/${TAG}/trained_models/${TRAINING}-${DATA}" \
    -p memory="25Gi" \
    -p cpu="4000m" \
    -p flags="--cache.local_download_path train-data-download-dir" \
    -p wandb-project="${WANDB_PROJECT}" \
    --name "${NAME}" \
    --labels "project=${PROJECT},experiment=${EXPERIMENT},trial=${TRIAL}"