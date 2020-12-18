#!/bin/bash

set -e

EXPERIMENT=2020-12-17-n2f-moisture-conservation
NSEGMENTS=20
RANDOM=$(openssl rand --hex 6)

argo submit --from workflowtemplate/prognostic-run \
    -p initial-condition="20160801.001500" \
    -p reference-restarts="gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts" \
    -p config="$(< nudge-to-fine-config.yaml)" \
    -p flags="--output-frequency 120" \
    -p segment-count="${NSEGMENTS}" \
    -p output="gs://vcm-ml-experiments/${EXPERIMENT}/moisture-conserving/nudge-to-fine-run" \
    --name "${EXPERIMENT}-nudge-to-fine-${RANDOM}"
