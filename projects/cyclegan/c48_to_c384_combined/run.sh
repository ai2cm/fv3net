#!/bin/bash

set -e

BUCKET=vcm-ml-experiments # don't pass bucket arg to use default 'vcm-ml-experiments'
PROJECT=cyclegan # don't pass project arg to use default 'default'
# TAG is the primary way by which users query for experiments with artifacts
# it determines the output directory for the data

EXPERIMENT="cyclegan_c48_to_c384_combined"
TRIAL="prec-lr-1e-5-decay-0.89125"
TAG=${EXPERIMENT}-${TRIAL}  # required
NAME=train-cyclegan-$(openssl rand --hex 6) # required

argo submit --from workflowtemplate/training-torch \
    -p output=$( artifacts resolve-url $BUCKET $PROJECT $TAG) \
    -p tag=${TAG} \
    -p training_config="$( yq . training.yaml )" \
    -p training_data_config="$( yq . train-data.yaml )" \
    -p validation_data_config="$( yq . validation-data.yaml )" \
    -p wandb-project="cyclegan_c48_to_c384" \
    -p cpu=4 \
    -p memory="15Gi" \
    --name "${NAME}" \
    --labels "project=${PROJECT},experiment=${EXPERIMENT},trial=${TRIAL}"

echo "argo job name: ${NAME}"
