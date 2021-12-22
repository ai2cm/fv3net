#!/bin/bash

set -e
set -o pipefail

if [[ "$1" == "--test" ]]; then
    extra_flags="--nfiles 2 --nfiles_valid 2 --epochs 5"
    bucket="vcm-ml-scratch"
    run_group="testing"
else
    bucket="vcm-ml-experiments"
    run_group="rnn-optimize-tests"
fi

group="$(openssl rand -hex 3)"
tag=":65382f5cd62c112f3e3cd2b7bc77e85d0aaa20b1"
argofile="../../train/argo.yaml"

submit () {
    model_name=$1
    flags=$2

    if [[ "$3" == "--gpu" ]]; then
        gpu_param="-p gpu-train=true"
    fi

    out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}-${group}")
    argo submit ${argofile} \
        --name "${model_name}-${group}" \
        -p wandb-run-group=${run_group} \
        -p tag="${tag}" \
        -p training-config="$(base64 --wrap 0 rnn-train.yaml)" \
        -p flags="--out_url ${out_url} ${flags} ${extra_flags}" \
        ${gpu_param}
}

# LR
# submit rnn-v1-optimize-lr-001 "--loss.optimizer.kwargs.learning_rate 0.001"

# batch_size
# submit rnn-v1-optimize-batch-256 "--loss.optimizer.kwargs.learning_rate 0.00014 --batch_size 512"
# submit rnn-v1-optimize-batch-512 "--loss.optimizer.kwargs.learning_rate 0.0002 --batch_size 512"
# submit rnn-v1-optimize-batch-1024 "--loss.optimizer.kwargs.learning_rate 0.00028 --batch_size 1024"

# GPU node w/ batch size
submit rnn-v1-optimize-gpu-batch-512 "--loss.optimizer.kwargs.learning_rate 0.0002 --batch_size 512" "--gpu"
# submit rnn-v1-optimize-gpu-batch-1024 "--loss.optimizer.kwargs.learning_rate 0.00028 --batch_size 1024" "--gpu"
