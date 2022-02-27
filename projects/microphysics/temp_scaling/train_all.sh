#!/bin/bash

set -e
set -o pipefail

if [[ "$1" == "--test" ]]; then
    extra_flags="--nfiles 10 --nfiles_valid 10 --epochs 2"
    bucket="vcm-ml-scratch"
else
    bucket="vcm-ml-experiments"
fi

tag="9a68fb66d235223056812cc302fe0c2fe2717a53"
group="$(openssl rand -hex 3)"


for exp in no-scaling qc qc-q all
do
for arch in dense rnn
do
    config_file="${exp}/${arch}.yaml"
    model_name="tscale-ablat-${exp}-${arch}-${group}"
    out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}")
    flags="--out_url ${out_url} ${extra_flags}"

    argo submit ../train/argo.yaml \
        --name "${model_name}" \
        -p training-config="$(base64 --wrap 0 $config_file)" \
        -p flags="$flags" \
        -p wandb-run-group="tscale-ablate-feb-2022-v2" \
        -p tag="${tag}"
done
done
