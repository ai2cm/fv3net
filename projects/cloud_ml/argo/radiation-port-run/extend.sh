#!/bin/bash

set -e

URL=$1

NAME="extend-cloud-ml-training-data-$(openssl rand --hex 2)"

argo submit --from workflowtemplate/restart-prognostic-run \
    -p url="$URL" \
    -p segment-count="2" \
    -p memory="20Gi" \
    --name="$NAME"