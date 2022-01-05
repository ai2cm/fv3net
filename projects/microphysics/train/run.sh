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
config=log-cloud
config_file="${config}.yaml"

for qcweight in 0.0 0.1 1.0 10.0 ; do
    model_name="${config}-residual-qcweight${qcweight}-${group}"
    out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}")

    argo submit argo.yaml \
        --name "${model_name}" \
        -p training-config="$(base64 --wrap 0 $config_file)" \
        -p flags="--loss.weights.cloud_water_mixing_ratio_after_precpd ${qcweight} \
        --out_url ${out_url} ${extra_flags}" | tee -a submitted-jobs.yaml
done
