#!/bin/bash

set -e

# adjust experiment name and replace "vcm-ml-scratch" with "vcm-ml-experiments"
# under output-url after debugging configuration
EXPERIMENT=2020-10-30-nudge-to-obs-GRL-paper/nudge-to-obs-run
IMAGE_TAG=12b6689ee53d1103ece4b8944ad8546861c2e1d6

gsutil cp diag_table gs://vcm-ml-experiments/diag_tables/nudge_to_obs_5h/v1.2/diag_table
argo submit \
    --from workflowtemplate/nudge-to-obs \
    -p fv3net-image=us.gcr.io/vcm-ml/fv3net:${IMAGE_TAG} \
    -p post-process-image=us.gcr.io/vcm-ml/post_process_run:${IMAGE_TAG} \
    -p fv3gfs-image=us.gcr.io/vcm-ml/prognostic_run:${IMAGE_TAG} \
    -p output-url=gs://vcm-ml-experiments/$EXPERIMENT \
    -p nudging-config="$(< nudge-to-obs-run.yaml)" \
    -p cpu="24" \
    -p memory="20Gi" \
    -p segment-count=37