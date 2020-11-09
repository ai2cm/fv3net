#!/bin/bash

set -e

# adjust experiment name and replace "vcm-ml-scratch" with "vcm-ml-experiments"
# under output-url after debugging configuration
EXPERIMENT=2020-10-30-nudge-to-obs-GRL-paper/nudge-to-obs-run-3hr-diags

gsutil cp diag_table gs://vcm-ml-experiments/diag_tables/nudge_to_obs_3h/v1.1/diag_table
argo submit \
    --from workflowtemplate/nudge-to-obs \
    -p output-url=gs://vcm-ml-experiments/$EXPERIMENT \
    -p nudging-config="$(< nudge-to-obs-run.yaml)" \
    -p cpu="24" \
    -p memory="20Gi" \
    -p segment-count="24" \
    -p flags="--python-output-interval 5"
