#!/bin/bash

set -e

BUCKET=vcm-ml-experiments # don't pass bucket arg to use default 'vcm-ml-experiments'
PROJECT=cyclegan # don't pass project arg to use default 'default'
# TAG is the primary way by which users query for experiments with artifacts
# it determines the output directory for the data

EXPERIMENT="cyclegan_c48_to_c384_combined"
TRIAL="prec-lr-1e-4-decay-0.63096-full-no-time-features-e13"
TAG=${EXPERIMENT}-${TRIAL}  # required
NAME=train-cyclegan-$(openssl rand --hex 6) # required
# TEMPLATE=workflowtemplate/training-torch

# PARAMETERS="-p output=$( artifacts resolve-url $BUCKET $PROJECT $TAG) \
#     -p training_config="$( yq . training.yaml )" \
#     -p training_data_config="$( yq . train-data.yaml )" \
#     -p validation_data_config="$( yq . validation-data.yaml )" \
#     -p wandb-project="cyclegan_c48_to_c384" \
#     -p cpu=4 \
#     -p memory=\"15Gi\""

# argo template get training-torch -o json | jq '[.spec.templates[] | select(.name == "training-torch") | .inputs.parameters[] | .name]' > template_parameters.json

# echo "$@" > submission_args.txt

# python3 validate_parameters.py $PARAMETERS

# rm submission_args.txt template_parameters.json

# echo $TEMPLATE
# echo $PARAMETERS
# argo submit --from $TEMPLATE \
#     $PARAMETERS \
#     --name "${NAME}" \
#     --labels "project=${PROJECT},experiment=${EXPERIMENT},trial=${TRIAL}"

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
