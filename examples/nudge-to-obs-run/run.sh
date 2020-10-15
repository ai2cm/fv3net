#!/bin/bash

set -e

# adjust experiment name and replace "vcm-ml-scratch" with "vcm-ml-experiments"
# under output-url after debugging configuration
EXPERIMENT=fill_in_here
IMAGE_TAG=4a8fc6aa3337d51967d5483695be582dd09561db

gsutil cp diag_table gs://vcm-ml-experiments/diag_tables/nudge_to_obs_5h/v1.0/diag_table
argo submit \
    --from workflowtemplate/nudge-to-obs \
    -p fv3net-image=us.gcr.io/vcm-ml/fv3net:${IMAGE_TAG} \
    -p post-process-image=us.gcr.io/vcm-ml/post_process_run:${IMAGE_TAG} \
    -p fv3gfs-image=us.gcr.io/vcm-ml/prognostic_run:${IMAGE_TAG} \
    -p output-url=gs://vcm-ml-scratch/$EXPERIMENT \
    -p nudging-config="$(< nudge-to-obs-run.yaml)" \
    -p cpu="24" \
    -p memory="20Gi" \
    -p segment-count=1