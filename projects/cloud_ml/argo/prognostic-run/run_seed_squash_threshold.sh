#!/bin/bash

set -e

CONFIG=$1
N_SEEDS=$2
SEGMENT_COUNT=$3
SQUASH_FILE=${4:-"squash_thresholds.yaml"}

PROJECT=cloud-ml
EXPERIMENT="cloud-ml-prog-run"
TRIAL="trial-0"

TMP_CONFIG="/tmp/tmp_config.yaml"

SQUASH_FILE="../../scripts/upload-squashed-models/${SQUASH_FILE}"
readarray -t SQUASH_THRESHOLDS < <(yq '.squash_thresholds[]' ${SQUASH_FILE})


for ((SEED=0;SEED<$N_SEEDS;SEED++))
do
    for SQUASH_THRESHOLD in "${SQUASH_THRESHOLDS[@]}"
    do
        TAG="${EXPERIMENT}-${CONFIG}-seed-${SEED}-${SQUASH_THRESHOLD}"
        NAME="${TAG}-$(openssl rand --hex 2)"

        echo "Preparing seed ${SEED} and squash threshold ${SQUASH_THRESHOLD}."
        cp "${CONFIG}-config.yaml" $TMP_CONFIG
        yq -yi --arg SEED "${SEED}" '.radiation_scheme.input_generator.model[] |= (sub("seed-0"; "seed-\($SEED)"))' ${TMP_CONFIG}
        yq -yi --arg SQUASH_THRESHOLD "${SQUASH_THRESHOLD}" \
            '.radiation_scheme.input_generator.model[] |= (sub("squash-threshold-0"; "squash-threshold-\($SQUASH_THRESHOLD | tonumber)"))' \
            ${TMP_CONFIG}

        argo submit --from workflowtemplate/prognostic-run \
            -p project=${PROJECT} \
            -p tag=${TAG} \
            -p config="$(< ${TMP_CONFIG})" \
            -p segment-count="${SEGMENT_COUNT}" \
            -p cpu="24" \
            -p memory="25Gi" \
            --name "${NAME}" \
            --labels "project=${PROJECT},experiment=${EXPERIMENT},trial=${TRIAL}"

        rm $TMP_CONFIG
    done
done