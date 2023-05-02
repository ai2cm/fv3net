#!/bin/bash

SWEEP_CONFIG=$1
TRAINING_CONFIG=$2

NAME=sweep-${SWEEP_CONFIG}-$(openssl rand --hex 2)

sleep 5

argo submit --from workflowtemplate/wandb-sweep \
    -p sweep_id="ai2cm/radiation-cloud-ml/${SWEEP_ID}" \
    -p training_config="$(< ./training_configs/${TRAINING_CONFIG}-training-config.json)" \
    -p training_data_config_remote_store="$(< ../training/train_small.yaml)" \
    -p training_data_config_local_batches="$(< ./data_configs/training-data-local-batches.yaml)" \
    -p validation_data_config_remote_store="$(< ../training/validation.yaml)" \
    -p validation_data_config_local_batches="$(< ./data_configs/validation-data-local-batches.yaml)" \
    -p wandb-project="radiation-cloud-ml" \
    --name $NAME

echo "argo get $NAME"