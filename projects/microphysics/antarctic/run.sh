#!/bin/bash

set -e
set -o pipefail

if [[ "$1" == "--test" ]]; then
    extra_flags="--nfiles 10 --nfiles_valid 10 --epochs 3"
    bucket="vcm-ml-scratch"
else
    bucket="vcm-ml-experiments"
fi

group="$(openssl rand -hex 3)"

# for config in rnn-cloudtdep dense-cloudtdep
for config in dense-cloudtdep
do
# for antarctic_only in true false
for antarctic_only in true
do
    config_file="${config}.yaml"
    model_name="zcemu-${config}-${group}-antarctic-${antarctic_only}"
    out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}")
    flags="--out_url ${out_url} ${extra_flags} \
    --transform.antarctic_only ${antarctic_only} \
    "

    argo submit ../train/argo.yaml \
        --name "${model_name}" \
        -p training-config="$(base64 --wrap 0 $config_file)" \
        -p wandb-run-group="antarctic-experiments-feb-2022" \
        -p flags="$flags" | tee -a submitted-jobs.txt
done
done
