#!/bin/bash

set -e

argo submit \
    --from workflowtemplate/prognostic-run \
    -p output=gs://vcm-ml-experiments/noah/prognostic_runs/2020-10-26-triggered-regressor \
    -p trained-ml=gs://vcm-ml-archive/noah/emulator/2020-10-16-triggered-regressor \
    -p reference-restarts=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
    -p initial-condition="20160805.000000" \
    -p config="$(< prognostic-run.yaml)" \
    -p segment-count=4
