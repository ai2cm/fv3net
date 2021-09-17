#!/bin/bash

mypy --follow-imports silent \
    external/vcm/vcm/ \
    workflows/dataflow/fv3net/pipelines/restarts_to_zarr/ \
    workflows/prognostic_c48_run \
    workflows/prognostic_c48_run/tests/ \
    external/fv3fit/fv3fit \
    external/loaders/loaders/ \
    workflows/diagnostics/fv3net/diagnostics/offline_ml_diags
