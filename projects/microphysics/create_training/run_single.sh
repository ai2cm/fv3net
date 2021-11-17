#!/bin/bash

set -e

PROG_YAML=$1
# to replace current_date in template
export MONTH=$2
OUTPUT_FREQUENCY=$3

IC_TIMESTAMP="2016${MONTH}01.000000"
# to replace initial_conditions in template
export IC_URL="gs://vcm-ml-raw/2020-11-05-GFS-month-start-initial-conditions-year-2016/2016${MONTH}0100"

envsubst < $PROG_YAML > "prognostic-run-with-IC.yaml"

# submit prognostic run forecasts
argo submit argo.yaml \
    -p config="$(< prognostic-run-with-IC.yaml)" \
    -p tag="create-training-${IC_TIMESTAMP}" \
    -p output_frequency=${OUTPUT_FREQUENCY}

rm prognostic-run-with-IC.yaml
