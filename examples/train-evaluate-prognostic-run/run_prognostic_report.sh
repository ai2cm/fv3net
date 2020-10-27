#!/bin/bash

set -e

EXPERIMENT=2020-10-26-n2f-timescale-sensitivity

argo submit \
    --from workflowtemplate/prognostic-run-diags \
    -p runs="$(< rundirs.json)" \
    -p docker-image=us.gcr.io/vcm-ml/fv3net:05b206107ed15178b97bc0dd3622e5918151b47f \
    -p make-movies="true" \
    --name "${EXPERIMENT}-prognostic-run-diags"