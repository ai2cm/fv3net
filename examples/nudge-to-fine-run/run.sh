#!/bin/bash

set -e

BUCKET=vcm-ml-scratch # don't pass bucket arg to use default 'vcm-ml-experiments'
PROJECT=brianh # don't pass project arg to use default 'default'
TAG=n2f-3km-timescale-6hr # required
NAME="${TAG}-nudge-to-fine-run-$(openssl rand --hex 6)"

argo submit --from workflowtemplate/prognostic-run \
    -p bucket=${BUCKET} \
    -p project=${PROJECT} \
    -p tag=${TAG} \
    -p config="$(< nudging-config.yaml)" \
    -p reference-restarts=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
    -p initial-condition="20160805.000000" \
    -p segment-count="1" \
    --name "${NAME}"
