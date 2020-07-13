#!/bin/bash

mypy --follow-imports silent \
    external/vcm/vcm/cloud \
    external/vcm/vcm/cubedsphere \
    fv3net/pipelines/restarts_to_zarr/ \
    workflows/prognostic_c48_run \
    fv3net/regression/sklearn \
    fv3net/regression/keras \
    external/loaders/loaders/mappers/_local.py \
    workflows/nudging
