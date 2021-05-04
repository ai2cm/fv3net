#!/bin/bash

set -e

BUCKET=vcm-ml-scratch # don't pass bucket arg to use default 'vcm-ml-experiments'
PROJECT=brianh # don't pass project arg to use default 'default'
TAG=n2o-test # required
NAME="${TAG}-nudge-to-obs-run-$(openssl rand --hex 6)"

argo submit --from workflowtemplate/prognostic-run \
    -p bucket=${BUCKET} \
    -p project=${PROJECT} \
    -p tag=${TAG} \
    -p config="$(< nudge-to-obs-run.yaml)" \
    -p initial-condition="20160801.000000" \
    -p reference-restarts=unused-parameter \
    -p cpu="24" \
    -p memory="25Gi" \
    -p segment-count=1 \
    --name "${NAME}"
