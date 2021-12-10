#!/bin/bash

set -e
set -o pipefail

if [[ "$1" == "--test" ]]; then
    extra_flags="--nfiles 2 --nfiles_valid 2 --epochs 10"
    bucket="vcm-ml-scratch"
else
    bucket="vcm-ml-experiments"
    extra_flags=""
fi

group="$(openssl rand -hex 3)"

for config in all-tendency-limited direct-cloud-limited; do
    for model_type in rnn-v1-shared-weights; do
        model_name="${config}-${model_type}"
        config_file="${config}.yaml"
        out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}-${group}")

        argo submit argo.yaml \
            --name "${model_name}-${group}" \
            -p wandb-run-group="rnn-fix-experiments" \
            -p training-config="$(base64 --wrap 0 $config_file)"\
            -p flags="--model.architecture.name ${model_type} --out_url ${out_url} ${extra_flags}"
    done
done
