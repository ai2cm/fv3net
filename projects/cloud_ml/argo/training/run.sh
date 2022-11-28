#!/bin/bash

set -e

BUCKET="vcm-ml-experiments"
PROJECT="cloud-ml"
EXPERIMENT="fine-cloud-dense-local"
TRIAL="trial-0"
TAG="${EXPERIMENT}"  # required
NAME=train-cloud-ml-$(openssl rand --hex 2) # required
WANDB_PROJECT="radiation-cloud-ml"

argo submit --from workflowtemplate/training \
    -p training_config="$( yq . training-config.yaml )" \
    -p training_data_config="$( yq . train.yaml )" \
    -p validation_data_config="$( yq . validation.yaml )" \
    -p output="gs://${BUCKET}/${PROJECT}/$(date +%F)/${TAG}/trained_model" \
    -p memory="20Gi" \
    -p flags="--cache.local_download_path train-data-download-dir" \
    -p wandb-project="${WANDB_PROJECT}" \
    --name "${NAME}" \
    --labels "project=${PROJECT},experiment=${EXPERIMENT},trial=${TRIAL}"