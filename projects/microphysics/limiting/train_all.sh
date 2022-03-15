#!/bin/bash

set -e
set -o pipefail

if [[ "$1" == "--test" ]]; then
    extra_flags="--nfiles 10 --nfiles_valid 10 --epochs 2"
    bucket="vcm-ml-scratch"
else
    bucket="vcm-ml-experiments"
fi

tag="dc38dde43dbb8b86450701c4c3e2f38355e1e8be"
group="$(openssl rand -hex 3)"


for exp in limiter limiter-all-loss
do
for arch in dense-no-tscale
do
    config_file="${exp}/${arch}.yaml"
    model_name="limit-tests-${exp}-${arch}-${group}"
    out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}")
    flags="--out_url ${out_url} ${extra_flags}"

    argo submit ../train/argo.yaml \
        --name "${model_name}" \
        -p training-config="$(base64 --wrap 0 $config_file)" \
        -p flags="$flags" \
        -p wandb-run-group="limiter-tests-feb-2022" \
        -p tag="${tag}"
done
done
