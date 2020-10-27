#!/bin/bash

set -e

EXPERIMENT=2020-10-26-n2f-timescale-sensitivity
TIMESCALES=(1, 3, 6, 10)
RANDOM=$(openssl rand --hex 6)

for TIMESCALE in ${TIMESCALES[@]}; do
    echo "${TIMESCALE} hr timescale"
    argo submit \
        --from workflowtemplate/train-diags-prog \
        -p image-tag="05b206107ed15178b97bc0dd3622e5918151b47f" \
        -p root="gs://vcm-ml-experiments/$EXPERIMENT/timescale-${TIMESCALE}hr" \
        -p train-test-data="gs://vcm-ml-experiments/${EXPERIMENT}/nudge-to-fine-runs/timescale-${TIMESCALE}hr" \
        -p training-config="$(< training-config.yaml)" \
        -p reference-restarts="gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts" \
        -p initial-condition="20160805.000000" \
        -p prognostic-run-config="$(< prognostic-run.yaml)" \
        -p train-times="$(<  ../../train.json)" \
        -p test-times="$(<  ../../test.json)" \
        -p public-report-output="gs://vcm-ml-public/offline_ml_diags/$EXPERIMENT/timescale-${TIMESCALE}hr" \
        -p segment-count=5 \
        --name "${EXPERIMENT}-train-diags-prog-${RANDOM}"
done ;
        

