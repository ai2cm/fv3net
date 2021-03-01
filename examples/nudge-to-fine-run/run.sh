#!/bin/bash

set -e

EXPERIMENT=2020-10-26-n2f-timescale-sensitivity
RANDOM=$(openssl rand --hex 6)

argo submit --from workflowtemplate/prognostic-run \
    -p output="gs://vcm-ml-scratch/oliwm/${EXPERIMENT}-test" \
    -p config="$(< nudging-config.yaml)" \
    -p reference-restarts=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
    -p initial-condition="20160805.000000" \
    -p segment-count="1" \
    --name "${EXPERIMENT}-nudge-to-fine-${RANDOM}"
