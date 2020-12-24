#!/bin/bash

set -e

EXPERIMENT=2020-12-17-n2f-moisture-conservation
NSEGMENTS=19
RANDOM=$(openssl rand --hex 6)

argo submit --from workflowtemplate/prognostic-run \
    -p initial-condition="20160801.010000" \
    -p config="$(< baseline-config.yaml)" \
    -p reference-restarts="gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts" \
    -p flags="--output-frequency 60" \
    -p output="gs://vcm-ml-experiments/${EXPERIMENT}/baseline/prognostic-run" \
    -p segment-count="${NSEGMENTS}" \
    -p chunks="$(< chunks.yaml)" \
    --name "${EXPERIMENT}-nudge-to-fine-${RANDOM}"
