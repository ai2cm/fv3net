#!/bin/bash

set -e
set -o pipefail

if [[ "$1" == "--test" ]]; then
    extra_flags="--nfiles 10 --nfiles_valid 10 --epochs 2"
    bucket="vcm-ml-scratch"
else
    bucket="vcm-ml-experiments"
fi

tag="35ace1cda7a18e47f064a9788dfe70fbea747b6c"
group="$(openssl rand -hex 3)"


for use_gen in true false
do
for buffer in null 13824
do

    model_name="tfdataset-gen-${use_gen}-buffer-${buffer}-${group}"
    out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}")
    flags="--out_url ${out_url} ${extra_flags}"

    export USE_GENERATOR=$use_gen
    export SHUFFLE_BUFFER_SIZE=$buffer
    envsubst dense.yaml > train.yaml
    trap "rm train.yaml" EXIT

    argo submit ../train/argo.yaml \
        --name "${model_name}" \
        -p training-config="$(base64 --wrap 0 train.yaml)" \
        -p flags="$flags" \
        -p wandb-run-group="sample-buffer-generator_or_map-mar-2022" \
        -p tag="${tag}"
done
done
