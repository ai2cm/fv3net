#!/bin/bash

set -e

EXPERIMENT=2020-12-17-n2f-moisture-conservation
NSEGMENTS=20
RANDOM=$(openssl rand --hex 6)

argo submit --from workflowtemplate/prognostic-run \
    -p initial-condition="20160801.001500" \
    -p config="$(< nudge-to-fine-config.yaml)" \
    -p reference-restarts="gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts" \
    -p flags="--output-frequency 120" \
    -p output="gs://vcm-ml-experiments/${EXPERIMENT}/moisture-conserving/nudge-to-fine-run" \
    -p segment-count="${NSEGMENTS}" \
    -p chunks="$(< chunks.yaml)" \
    --name "${EXPERIMENT}-nudge-to-fine-${RANDOM}"
