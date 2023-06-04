#!/bin/bash

set -e
set -x

CONFIG=$1
SEGMENT_COUNT=$2
N_SEEDS=${3:-5}

PROJECT=cloud-ml
EXPERIMENT="cloud-ml-prog-run"
TRIAL="trial-0"

TMP_CONFIG=tmp-config.yaml

for ((SEED=0;SEED<=$N_SEEDS;SEED++))
do
    TAG=${EXPERIMENT}-${CONFIG}-seed-$SEED
    NAME="${TAG}-$(openssl rand --hex 2)"

    echo "Preparing seed ${SEED}..."
    cp "${CONFIG}-config.yaml" $TMP_CONFIG
    sed -i "s/seed-0/seed-$SEED/" $TMP_CONFIG

    argo submit --from workflowtemplate/prognostic-run \
        -p project=${PROJECT} \
        -p tag=${TAG} \
        -p config="$(< $TMP_CONFIG)" \
        -p segment-count="${SEGMENT_COUNT}" \
        -p cpu="24" \
        -p memory="25Gi" \
        --name "${NAME}" \
        --labels "project=${PROJECT},experiment=${EXPERIMENT},trial=${TRIAL}"

    rm $TMP_CONFIG
done