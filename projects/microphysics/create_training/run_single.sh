#!/bin/bash

set -e

PROG_YAML=$1
MONTH=$2
PROJECT=$3

RANDOM="$(openssl rand --hex 4)"
IC_TIMESTAMP="2016${MONTH}01.000000"
export IC_URL="gs://vcm-ml-raw/2020-11-05-GFS-month-start-initial-conditions-year-2016/2016${MONTH}0100"
NC_OUTPUT_FREQ_SEC=18000

envsubst < $PROG_YAML > "prognostic-run-with-IC.yaml"

# submit prognostic run forecasts
argo submit \
    --from workflowtemplate/prognostic-run \
    --name create-emu-micro-data-${MONTH}-${RANDOM} \
    -p reference-restarts=null \
    -p initial-condition=$IC_TIMESTAMP \
    -p config="$(< prognostic-run-with-IC.yaml)" \
    -p memory="10Gi" \
    -p cpu="6" \
    -p project=${PROJECT} \
    -p tag="init-${IC_TIMESTAMP}" \
    -p online-diags="false" \
    -p output_freq_sec=${NC_OUTPUT_FREQ_SEC}

rm prognostic-run-with-IC.yaml