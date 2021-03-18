#!/bin/bash

set -e

# adjust experiment name and replace "vcm-ml-scratch" with "vcm-ml-experiments"
# under output-url after debugging configuration
EXPERIMENT=fill_in_here

argo submit \
    --from workflowtemplate/prognostic-run \
    -p output=gs://vcm-ml-scratch/$EXPERIMENT \
    -p config="$(< nudge-to-obs-run.yaml)" \
    -p initial-condition="20160801.000000" \
    -p reference-restarts=unused-parameter \
    -p cpu="24" \
    -p memory="25Gi" \
    -p segment-count=1
