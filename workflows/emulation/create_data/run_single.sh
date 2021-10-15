#!/bin/bash
set -e

MONTH=$(printf '%02d' $1)
RANDOM="$(openssl rand --hex 4)"
export IC_URL="gs://vcm-ml-raw/2020-11-05-GFS-month-start-initial-conditions-year-2016/2016${MONTH}0100"

# submit prognostic run forecasts
# even though these runs are initialized with GFS reference-restarts and
# initial-conditions must be provided for compatibility with prepare-config
argo submit \
    --from workflowtemplate/prognostic-run \
    --name all-physics-emu-train-${MONTH}-${RANDOM} \
    -p project=online-emulator \
    -p tag=gfs-initialized-baseline-$MONTH \
    -p reference-restarts=null \
    -p initial-condition="2016${MONTH}01.000000" \
    -p config="$(envsubst < fv3config.yaml)" \
    -p memory="10Gi" \
    -p cpu="6"