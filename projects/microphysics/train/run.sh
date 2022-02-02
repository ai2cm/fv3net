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

for lr in 0.0002
do
for ch in 256 512
do
    model_name="rnn-alltdep-${group}-de${ch}-lr${lr}-login"
    out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}")
    flags="--out_url ${out_url} ${extra_flags} \
    --loss.optimizer.kwargs.learning_rate $lr \
    --model.architecture.kwargs.channels $ch\
    "
    argo submit argo.yaml \
        --name "${model_name}" \
        -p training-config="$(base64 --wrap 0 $config_file)" \
        -p flags="$flags" | tee -a submitted-jobs.txt
done
done
