#!/bin/bash

mypy --follow-imports silent \
    external/vcm/vcm/cloud \
    external/vcm/vcm/cubedsphere \
    fv3net/pipelines/restarts_to_zarr/ \
    workflows/prognostic_c48_run \
    external/fv3fit/fv3fit \
    external/loaders/loaders/mappers/_local.py
