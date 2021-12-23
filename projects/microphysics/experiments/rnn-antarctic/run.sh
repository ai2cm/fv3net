#!/bin/bash

set -e
set -o pipefail

if [[ "$1" == "--test" ]]; then
    extra_flags="--nfiles 2 --nfiles_valid 2 --epochs 5"
    bucket="vcm-ml-scratch"
else
    bucket="vcm-ml-experiments"
fi

group="$(openssl rand -hex 3)"
tag=":4476a8f50bb2e856cfa6d165c0e84abcd44df6e3"

submit () {
    model_name=$1
    flags=$2

    out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}-${group}")
    argo submit ../../train/argo.yaml \
        --name "${model_name}-${group}" \
        -p wandb-run-group="antarctic-training-experiment" \
        -p tag="${tag}" \
        -p training-config="$(base64 --wrap 0 rnn-v1-shared-weights.yaml)" \
        -p flags="--out_url ${out_url} ${flags} ${extra_flags}" \
        -p gpu-train="true"
}

submit direct-qc-rnn-v1-shared-weights-antarctic-only
