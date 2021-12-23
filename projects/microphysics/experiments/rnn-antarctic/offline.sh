#!/bin/bash

set -e
set -o pipefail

argo submit ../../argo/prog-run-and-eval.yaml \
    -p tag="direct-qc-rnn-antarctic-only-86560b" \
    -p image_tag="ada69b948dcb3eee8416c91e2c1efcc099cac5bf" \
    -p baseline_tag="baseline-short" \
    -p config="$(base64 --wrap 0  ../../configs/default_short.yaml)" \
    -p tf_model="gs://vcm-ml-experiments/microphysics-emulation/2021-12-22/direct-qc-rnn-v1-shared-weights-antarctic-only-86560b/model.tf"
