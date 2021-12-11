#!/bin/bash
set -e

arch=rnn-v1-shared-weights
nfiles=200
cache=true
checkpoint=true

if [[ "$1" == "--docker" ]]; then
    docker run -v $(pwd)/../../../:/fv3net -w /fv3net/projects/microphysics/argo  \
        --rm \
        us.gcr.io/vcm-ml/prognostic_run:3136fac543fd15e8b7d70b5db03573a8fe6c1b98 \
        bash run_memory_leak.sh
    sudo chown -R noahb:noahb .
else
    ROOT=../../..
    pip install memory-profiler
    pip install $ROOT/external/fv3fit
    # mprof run python3 memory_leak.py
    export WANDB_MODE=offline
    tmp=$(mktemp -d)/profile.dat
    set +e
    mprof run --output $tmp python3 -m fv3fit.train_microphysics --config-path ../train/direct-cloud-all-levs-conservative.yaml --out_url local1 --epochs 1 --nfiles $nfiles \
        --conservative_model.architecture.name $arch \
        --checkpoint_model  $checkpoint \
        --use_wandb false \
        --nfiles_valid $nfiles --verbose 1 --log_level DEBUG
    mprof plot --output $arch-check$checkpoint-cache$cache-$(hostname)-nfiles${nfiles}-$(date -Is).png $tmp
fi
