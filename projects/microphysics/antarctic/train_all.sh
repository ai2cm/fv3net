#!/bin/bash

set -e
set -o pipefail

if [[ "$1" == "--test" ]]; then
    extra_flags="--nfiles 10 --nfiles_valid 10 --epochs 2"
    bucket="vcm-ml-scratch"
else
    bucket="vcm-ml-experiments"
fi

tag="011435e1b91e21c14d610abcc35150548e063a9b"
group="$(openssl rand -hex 3)"


for exp in cloud-tdep-gscond-cbfc4a alltdep-log-gscond-limited
do
for arch in rnn
do
for ant in true false
do
    config_file="${exp}/${arch}.yaml"
    model_name="${exp}-${arch}-antarc-${ant}-${group}"
    out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}")
    flags="--out_url ${out_url} ${extra_flags} --transform.antarctic_only ${ant}"
    argo submit ../train/argo.yaml \
        --name "${model_name}" \
        -p training-config="$(base64 --wrap 0 $config_file)" \
        -p flags="$flags" \
        -p wandb-run-group="antarctic-experiments-v2-feb-2022" \
        -p tag="${tag}"
done
done
done
