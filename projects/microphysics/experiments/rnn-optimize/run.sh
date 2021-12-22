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
tag=":c493e95fe3b0f2df24002d95601550eb62383ede"
argofile="../../train/argo.yaml"

submit () {
    model_name=$1
    config=$2
    flags=$3

    if [[ "$4" == "--gpu" ]]; then
        gpu_param="-p gpu-train=true"
    fi

    out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}-${group}")
    argo submit ${argofile} \
        --name "${model_name}-${group}" \
        -p wandb-run-group=${run_group} \
        -p tag="${tag}" \
        -p training-config="$(base64 --wrap 0 ${config})" \
        -p flags="--out_url ${out_url} ${flags} ${extra_flags}" \
        ${gpu_param}
}

# LR
# submit rnn-v1-optimize-lr-001 rnn-train.yaml "--loss.optimizer.kwargs.learning_rate 0.001"

# batch_size
# submit rnn-v1-optimize-batch-256 rnn-train.yaml "--loss.optimizer.kwargs.learning_rate 0.00014 --batch_size 512"
# submit rnn-v1-optimize-batch-512 rnn-train.yaml "--loss.optimizer.kwargs.learning_rate 0.0002 --batch_size 512"
# submit rnn-v1-optimize-batch-1024 rnn-train.yaml "--loss.optimizer.kwargs.learning_rate 0.00028 --batch_size 1024"

# GPU node w/ batch size
# submit rnn-v1-optimize-gpu-batch-512 rnn-train.yaml "--loss.optimizer.kwargs.learning_rate 0.0002 --batch_size 512" "--gpu"
# submit rnn-v1-optimize-gpu-batch-1024 rnn-train.yaml "--loss.optimizer.kwargs.learning_rate 0.00028 --batch_size 1024" "--gpu"

# GPU node w/ learning rate schedule
submit rnn-v1-optimize-gpu-batch-1024-lr-sched rnn-train-lr-schedule.yaml "--batch_size 1024" "--gpu"
# submit rnn-v1-optimize-gpu-batch-2048-lr-sched rnn-train-lr-schedule.yaml "--batch_size 1024 --loss.optimizer.learning_rate.initial_learning_rate 0.0004" "--gpu"

