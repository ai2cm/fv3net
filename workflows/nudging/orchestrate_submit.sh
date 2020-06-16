#!/bin/bash

set -e

usage="usage: orchestrate_submit.sh \
       base_nudging_config nudge_timescale_hrs fv3gfs_image output_root_path"

if [ $# -lt 4 ]; then
    echo -e $usage
    exit 1
fi

BASE_CONFIG=$1
NUDGE_TIMESCALE_HOURS=$2
FV3GFS_IMAGE=$3
REMOTE_ROOT=$4

RUN_LABEL=${NUDGE_TIMESCALE_HOURS}h
REMOTE_OUTDIR=${REMOTE_ROOT}/outdir-${RUN_LABEL}
RUNFILE=runfile.py
REMOTE_RUNFILE=${REMOTE_ROOT}/${RUNFILE}
CONFIG_FILE=fv3config-${RUN_LABEL}.yaml
REMOTE_CONFIG_FILE=${REMOTE_ROOT}/config/${CONFIG_FILE}


cd workflows/nudging

python prepare_config.py ${BASE_CONFIG} ${CONFIG_FILE} --timescale-hours ${NUDGE_TIMESCALE_HOURS}

gsutil cp ${CONFIG_FILE} ${REMOTE_CONFIG_FILE}
gsutil cp ${RUNFILE} ${REMOTE_RUNFILE}
./submit_job.sh ${REMOTE_CONFIG_FILE} ${REMOTE_RUNFILE} ${REMOTE_OUTDIR} ${FV3GFS_IMAGE}


