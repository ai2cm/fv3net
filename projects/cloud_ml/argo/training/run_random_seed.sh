#!/bin/bash

set -e
set -x

TRAINING_CONFIG=$1
N_SEEDS=$2

BUCKET="vcm-ml-experiments"
PROJECT="cloud-ml"
EXPERIMENT="train-cloud-ml-${TRAINING_CONFIG}"
TAG="${EXPERIMENT}"
WANDB_GROUP="${EXPERIMENT}"
WANDB_PROJECT="radiation-cloud-ml"

TMP_CONFIG="tmp-config.yaml"

for ((SEED=0;SEED<$N_SEEDS;SEED++))
do
    NAME="${TAG}-$(openssl rand --hex 2)"

    echo "Preparing seed ${SEED}..."
    cp "${TRAINING_CONFIG}-config.yaml" $TMP_CONFIG
    sed -i "s/random_seed: 0/random_seed: $SEED/" $TMP_CONFIG
    sed -i "s/seed-0/seed-$SEED/" $TMP_CONFIG


    argo submit --from workflowtemplate/training \
        -p training_config="$( yq . ${TMP_CONFIG})" \
        -p training_data_config="$( yq . train.yaml )" \
        -p validation_data_config="$( yq . validation.yaml )" \
        -p output="gs://${BUCKET}/${PROJECT}/$(date +%F)/${TAG}/trained_models/${TRAINING_CONFIG}-seed-${SEED}" \
        -p memory="25Gi" \
        -p cpu="4000m" \
        -p flags="--cache.local_download_path train-data-download-dir" \
        -p wandb-project="${WANDB_PROJECT}" \
        -p wandb-group="${WANDB_GROUP}" \
        --name "${NAME}" \
        --labels "project=${PROJECT},experiment=${EXPERIMENT},trial=trial-0"

    rm $TMP_CONFIG
done