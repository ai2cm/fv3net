#!/bin/bash

set -e

N_SEEDS=$1
OUTPUT_ROOT=$2
SQUASH_FILE=${3:-"squash_thresholds.yaml"}
MODEL_CONFIG=${4:-"squashed_output_model.yaml"}

TMP_CONFIG="/tmp/tmp_config.yaml"
readarray -t SQUASH_THRESHOLDS < <(yq '.squash_thresholds[]' ${SQUASH_FILE})

for ((SEED=0;SEED<$N_SEEDS;SEED++))
do
    for SQUASH_THRESHOLD in "${SQUASH_THRESHOLDS[@]}"
    do
        cp ${MODEL_CONFIG} ${TMP_CONFIG}
        yq -yi --arg SEED "${SEED}" '.base_model_path |= (sub("seed-0"; "seed-\($SEED)"))' ${TMP_CONFIG}
        yq -yi --arg SQUASH_THRESHOLD "${SQUASH_THRESHOLD}" '.squashing[].squash_threshold |= ($SQUASH_THRESHOLD | tonumber)' ${TMP_CONFIG}
        gsutil cp ${TMP_CONFIG} "${OUTPUT_ROOT}/squashed-models/dense-seed-${SEED}/squash-threshold-"$SQUASH_THRESHOLD"/squashed_output_model.yaml"
    done
done

rm $TMP_CONFIG