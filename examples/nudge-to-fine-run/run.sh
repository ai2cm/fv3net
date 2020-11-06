#!/bin/bash

set -e

EXPERIMENT=2020-10-26-n2f-timescale-sensitivity
NSEGMENTS=1
RANDOM=$(openssl rand --hex 6)

argo submit --from workflowtemplate/nudging \
    -p nudging-config="$(< nudging-config.yaml)" \
    -p segment-count="${NSEGMENTS}" \
    -p output-url="gs://vcm-ml-scratch/brianh/${EXPERIMENT}-test" \
    -p times="$(< output-times-test.json)" \
    --name "${EXPERIMENT}-nudge-to-fine-${RANDOM}"
