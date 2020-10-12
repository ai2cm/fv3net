#!/bin/bash

set -e
IMAGE_TAG=42e3dbcc5995c4dce9c71c93ff60c489ba0245b7
argo submit --from=workflowtemplates/prognostic-run-diags \
    -p docker-image=us.gcr.io/vcm-ml/fv3net:$IMAGE_TAG \
    -p runs="$(< rundirs.json)" \
    -p make-movies=false \
    --name prog-diags-nn-random-seed
