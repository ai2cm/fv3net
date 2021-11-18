#!/bin/bash

set -e
set -o pipefail

config_file=$1
model_type=$2
model_prefix=$3

model_name=${model_prefix}-${model_type}
out_url=$(artifacts resolve-url vcm-ml-experiments microphysics-emulation ${model_name})

argo submit argo.yaml \
    -p training-config="$(< $config_file)" \
    -p flags="--model.architecture.name ${model_type} --wandb_model_name ${model_name} --out_url ${out_url}"