#!/bin/bash

TRAINED_ML=/Users/noah/data/gs/vcm-ml-archive/noahb/hybrid-fine-res/trained_ml
OFFLINE_DIAGS=/Users/noah/data/gs/vcm-ml-archive/noahb/hybrid-fine-res/offline_diags

function train {
    config="$1"
    out="$2"

    if ! [[ -d $out ]]; then
        python -m fv3fit.train \
            --timesteps-file train.json \
            /Users/noah/data/gs/vcm-ml-archive/noahb/hybrid-fine-res/2021-05-05-hybrid-training.zarr \
            "$config" \
            "$out"
    fi
}

function evaluate {
    python -m offline_ml_diags.compute_diags \
        "$1" \
        "$2" \
        --timesteps-file test.json
}


# train training.yml $TRAINED_ML/full_atmos_input
# train training-lower.yml $TRAINED_ML/lower

mkdir -p $OFFLINE_DIAGS
for model in full_atmos_input lower
do
    if ! [ -d $OFFLINE_DIAGS/$model ]; then
        evaluate $TRAINED_ML/$model $OFFLINE_DIAGS/$model
    fi
done