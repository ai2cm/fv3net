#!/bin/bash

set -e

PROG_YAML=$1
MONTH=$2
OUTPUT_FREQUENCY=$3

IC_TIMESTAMP="2016${MONTH}01.000000"
IC_URL="gs://vcm-ml-raw/2020-11-05-GFS-month-start-initial-conditions-year-2016/2016${MONTH}0100"

# envsubst < $PROG_YAML > "prognostic-run-with-IC.yaml"

# submit prognostic run forecasts
argo submit argo.yaml \
    -p initial_condition=$IC_URL \
    -p config="$(< ${PROG_YAML})" \
    -p tag="create-training-${IC_TIMESTAMP}" \
    -p output_frequency=${OUTPUT_FREQUENCY}
