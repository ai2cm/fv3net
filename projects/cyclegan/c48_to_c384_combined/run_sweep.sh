#!/bin/bash

set -e

# Manually run the 'wandb sweep ...' command first to get the sweep ID, then paste into argo submit parameter field
# wandb sweep --entity ai2cm --project cyclegan-tuning sweep.yaml
# wandb sweep --entity annakwa --project test-pire-dense-model-tuning sweep.yaml

NAME=sweep-cyclegan-$(openssl rand --hex 6)

argo submit --from workflowtemplate/wandb-sweep-torch-v1 \
    -p sweep_id="ai2cm/cyclegan-tuning/g9wo6d38" \
    -p training_config="$(< training.yaml)" \
    -p training_data_config="$(< train-data.yaml)" \
    -p validation_data_config="$(< validation-data.yaml)" \
    -p max_runs="20" \
    --name $NAME

echo "argo get $NAME"
