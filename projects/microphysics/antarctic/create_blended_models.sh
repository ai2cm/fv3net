#!/bin/bash

set -e
set -o pipefail

if [[ "$1" == "--test" ]]; then
    bucket="vcm-ml-scratch"
else
    bucket="vcm-ml-experiments"
fi

group="$(openssl rand -hex 3)"
model_name="rnn-cloudtdep-cbfc4a-antarctic-blend-${group}"
out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}")

python3 create_blended.py \
    gs://vcm-ml-experiments/microphysics-emulation/2022-03-02/cloud-tdep-gscond-cbfc4a-rnn-antarc-true-a5b933/model.tf \
    gs://vcm-ml-experiments/microphysics-emulation/2022-03-02/cloud-tdep-gscond-cbfc4a-rnn-antarc-false-a5b933/model.tf \
    $out_url

# export WANDB_RUN_GROUP=antarctic-experiments-v2-feb-2022
# export WANDB_NAME=rnn-cloudtdep-cbfc4a-antarctic-blend-ae2a48
# out_url="gs://vcm-ml-experiments/microphysics-emulation/2022-03-05/rnn-cloudtdep-cbfc4a-antarctic-blend-ae2a48"

# python3 ../scripts/score_training.py \
#     --model_url ${out_url}/model.tf \
#     --config-path cloud-tdep-gscond-cbfc4a/rnn.yaml \
#     --wandb.job_type train_score

