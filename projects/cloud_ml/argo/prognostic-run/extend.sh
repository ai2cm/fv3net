#!/bin/bash

set -e

URL=$1
SEGMENT_COUNT=$2

NAME="cloud-ml-prognostic-run-$(openssl rand --hex 2)"

argo submit --from workflowtemplate/restart-prognostic-run \
    -p url="$URL" \
    -p segment-count="${SEGMENT_COUNT}" \
    -p cpu="24" \
    -p memory="25Gi" \
    --name="$NAME"