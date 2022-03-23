#!/bin/bash

NAME="control-rnn"
CONFIG_FILE=rnn.yaml

set -e
set -o pipefail

group="$(openssl rand -hex 3)"

model_name="$NAME-${group}"
out_url=$(artifacts resolve-url vcm-ml-experiments microphysics-emulation "${model_name}")
argo submit argo.yaml \
    --name "${model_name}" \
    -p training-config="$(base64 --wrap 0 $CONFIG_FILE)"\
    -p flags="--out_url ${out_url}" \
    | tee -a experiment-log.txt
