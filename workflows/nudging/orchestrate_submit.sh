#!/bin/bash

set -e
set -x

usage="usage: orchestrate_submit.sh \
       base_nudging_config nudge_timescale_hrs fv3gfs_image output_root_path pvc"

if [ $# -lt 5 ]; then
    echo -e $usage
    exit 1
fi

BASE_CONFIG=$1
NUDGE_TIMESCALE_HOURS=$2
FV3GFS_IMAGE=$3
REMOTE_ROOT=$4
PVC=$5

RUN_LABEL=${NUDGE_TIMESCALE_HOURS}h
REMOTE_OUTDIR=${REMOTE_ROOT}/outdir-${RUN_LABEL}
RUNFILE=runfile.py
CONFIG_FILE=fv3config-${RUN_LABEL}.yaml


cd workflows/nudging

python prepare_config.py ${BASE_CONFIG} ${CONFIG_FILE} --timescale-hours ${NUDGE_TIMESCALE_HOURS}
./submit_job.sh ${CONFIG_FILE} $RUNFILE  ${REMOTE_OUTDIR} ${FV3GFS_IMAGE} $PVC nudging-output-1


