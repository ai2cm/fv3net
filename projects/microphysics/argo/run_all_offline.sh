#!/bin/bash

set -e
set -x

BASE_TAG="prognostic-offline"
RANDOM=$(openssl rand --hex 4)
MODEL_BASE="gs://vcm-ml-experiments/2021-10-14-microphsyics-emulation-paper/models"

run_style=("offline", "online")
qc_tends=("all-tends-limted" "direct-qc-limited")
archictectures=("dense", "rnn")

for flag in ${run_style[@]}; do
    for tend in ${qc_tends[@]}; do
        for arch in ${archictectures[@]}; do
            TAG=${BASE_TAG}-${tend}-${arch}-${RANDOM}
            MODEL=${MODEL_BASE}/${tend}/${tend}-${arch}/model.tf
            argo submit prog-run-and-eval.yaml \
                -p tag=$TAG \
                -p config=$(< offline.yaml) \
                -p tf_model=$MODEL \
                -p on_off_flag="--${flag}"
        done
    done
done
