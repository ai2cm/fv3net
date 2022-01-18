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
config=rnn
config_file="${config}.yaml"

model_name="${config}-predict-gscond-${group}"
out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}")

argo submit argo.yaml \
    --name "${model_name}" \
    -p training-config="$(base64 --wrap 0 $config_file)" \
    -p flags="--out_url ${out_url} ${extra_flags}" | tee -a submitted-jobs.txt
