#!/bin/bash

set -e

INITIAL_MONTH=$1
OUTPUT=gs://vcm-ml-experiments/2020-10-30-nudge-to-obs-GRL-paper/weather-forecasts/$INITIAL_CONDITION
ML_MODEL=gs://vcm-ml-experiments/2020-10-30-nudge-to-obs-GRL-paper/rf-default/trained_model

IC_TIMESTAMP="2016${INITIAL_MONTH}01.000000"
IC_URL="gs://vcm-ml-raw/2020-11-05-GFS-month-start-initial-conditions-year-2016/2016${INITIAL_MONTH}0100"

envsubst < "prognostic-run.yaml" > "prognostic-run-with-IC.yaml"

argo submit \
    --from workflowtemplate/prognostic-run \
    -p output="${OUTPUT}-prognostic" \
    -p trained-ml=$ML_MODEL \
    -p reference-restarts=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
    -p initial-condition=$IC_TIMESTAMP \
    -p config="$(< prognostic-run-with-IC.yaml)"
    -p flags="--nudge-to-observations"


argo submit \
    --from workflowtemplate/prognostic-run \
    -p output="${OUTPUT}-baseline" \
    -p trained-ml=$ML_MODEL \
    -p reference-restarts=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
    -p initial-condition=$IC_TIMESTAMP \
    -p config="$(< prognostic-run-with-IC.yaml)" \
    -p flags="--nudge-to-observations --diagnostic_ml"