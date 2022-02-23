#!/bin/bash

set -e
set -o pipefail

if [[ "$1" == "--test" ]]; then
    extra_flags="--nfiles 10 --nfiles_valid 10 --epochs 2"
    bucket="vcm-ml-scratch"
else
    bucket="vcm-ml-experiments"
fi

tag="1d6db454e5cd9c77f07115f121244ca5f416dfc6"
group="$(openssl rand -hex 3)"


for exp in base gscond
do
for arch in dense rnn
do
    config_file="${exp}/${arch}.yaml"
    model_name="in-ablat-${exp}-${arch}-${group}"
    out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}")
    flags="--out_url ${out_url} ${extra_flags}"

    argo submit ../train/argo.yaml \
        --name "${model_name}" \
        -p training-config="$(base64 --wrap 0 $config_file)" \
        -p flags="$flags" \
        -p wandb-run-group="input-ablation-feb-2022" \
        -p tag="${tag}"
done
done
