#!/bin/bash

set -e
set -o pipefail

for config in all-tendency-limited direct-cloud-limited; do
    for model_type in rnn linear dense; do
        model_name="${config}-${model_type}"
        config_file="${config}.yaml"
        out_url=$(artifacts resolve-url vcm-ml-experiments microphysics-emulation "${model_name}")

        argo submit argo.yaml \
            -p training-config="$(base64 --wrap 0 $config_file)"\
            -p flags="--model.architecture.name ${model_type} --wandb_model_name ${model_name} --out_url ${out_url}"

    done
done
