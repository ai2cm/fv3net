#!/bin/bash

set -e
set -o pipefail

if [[ "$1" == "--test" ]]; then
    extra_flags="--nfiles 10 --nfiles_valid 10 --epochs 2"
    bucket="vcm-ml-scratch"
else
    bucket="vcm-ml-experiments"
fi

group="$(openssl rand -hex 3)"


# for config in rnn-control
for config in rnn-control rnn-limited-qc rnn
do
    config_file="${config}.yaml"
    model_name="dqc-precpd-limiter-${config}-${group}"
    out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}")
    flags="--out_url ${out_url} ${extra_flags}"

    argo submit ../train/argo.yaml \
        --name "${model_name}" \
        -p training-config="$(base64 --wrap 0 $config_file)" \
        -p flags="$flags" \
        -p wandb-run-group="qc-precpd-limiter-v2-mar2022"
done
