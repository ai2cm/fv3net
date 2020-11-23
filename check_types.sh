#!/bin/bash

mypy --follow-imports silent \
    external/vcm/vcm/cloud \
    external/vcm/vcm/cubedsphere \
    workflows/dataflow/fv3net/pipelines/restarts_to_zarr/ \
    workflows/prognostic_c48_run \
    workflows/prognostic_c48_run/tests/ \
    external/fv3fit/fv3fit \
    external/loaders/loaders/mappers/_local.py \
    external/loaders/loaders/batches/_sequences.py \
    external/vcm/vcm/derived_mapping.py \
    workflows/offline_ml_diags/offline_ml_diags