#!/bin/bash

set -e
set -o pipefail

if [[ "$1" == "--test" ]]; then
    bucket="vcm-ml-scratch"
else
    bucket="vcm-ml-experiments"
fi

model_name="rnn-alltdep-antarctic-blend"
out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}")

python3 create_blended.py \
    gs://vcm-ml-experiments/microphysics-emulation/2022-02-10/zcemu-rnn-cloudtdep-b0cc65-antarctic-true/model.tf \
    gs://vcm-ml-experiments/microphysics-emulation/2022-02-10/zcemu-rnn-cloudtdep-b0cc65-antarctic-false/model.tf \
    $out_url

model_name="dense-alltdep-antarctic-blend"
out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}")

python3 create_blended.py \
    gs://vcm-ml-experiments/microphysics-emulation/2022-02-10/zcemu-dense-cloudtdep-b0cc65-antarctic-true/model.tf \
    gs://vcm-ml-experiments/microphysics-emulation/2022-02-10/zcemu-dense-cloudtdep-b0cc65-antarctic-false/model.tf \
    $out_url