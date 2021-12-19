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
tag=":acc03550cc1b9148b806f11e01f9ff01b1e2f87a"

submit () {
    model_name=$1
    flags=$2

    if [[ "$3" == "--gpu" ]]; then
        argofile="argo_gpu.yaml"
        tag=":cuda-base-v2"
    else
        argofile="../../train/argo.yaml"
    fi

    out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}-${group}")
    argo submit ${argofile} \
        --name "${model_name}-${group}" \
        -p wandb-run-group="rnn-optimize-tests" \
        -p tag="${tag}" \
        -p training-config="$(base64 --wrap 0 rnn-train.yaml)" \
        -p flags="--out_url ${out_url} ${flags} ${extra_flags}" \
        ${node}
}

# LR
submit rnn-v1-optimize-lr-001 "--loss.optimizer.kwargs.learning_rate 0.001"

# batch_size
submit rnn-v1-optimize-batch-256 "--loss.optimizer.kwargs.learning_rate 0.00014 --batch_size 512"
submit rnn-v1-optimize-batch-512 "--loss.optimizer.kwargs.learning_rate 0.0002 --batch_size 512"
submit rnn-v1-optimize-batch-1024 "--loss.optimizer.kwargs.learning_rate 0.00028 --batch_size 1024"

# GPU node w/ batch size
submit rnn-v1-optimize-gpu-batch-512 "--loss.optimizer.kwargs.learning_rate 0.0002 --batch_size 512" "--gpu"
submit rnn-v1-optimize-gpu-batch-1024 "--loss.optimizer.kwargs.learning_rate 0.00028 --batch_size 1024" "--gpu"
