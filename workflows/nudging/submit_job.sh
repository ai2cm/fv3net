#!/bin/bash

set -e

usage="usage: submit_job.sh [-h] [-j JOB_PREFIX] \
       [-d] config runfile output_url docker_image \n\
       -j JOB_PREFIX: job name prefix for k8s submission \n\
       -d: detach job from terminal session" 
detach=0
job_prefix="nudge-to-highres"

while getopts "j:dh" OPTION; do
    case $OPTION in
        j)
            job_prefix=$OPTARG
        ;;
        h)
            echo -e $usage
            exit 1
        ;;
        d)
            detach=1
        ;;
        *)
            echo -e $usage
            exit 1
        ;;
    esac
done

shift $(($OPTIND - 1))

if [ $# -lt 4 ]; then
    echo -e $usage
    exit 1
fi

rand_tag=$(openssl rand --hex 4)

export CONFIG=$1
export RUNFILE=$2
export OUTPUT_URL=$3
export DOCKER_IMAGE=$4
export JOBNAME=$job_prefix-$rand_tag

cat job_template.yaml | \
    envsubst '$CONFIG $RUNFILE $OUTPUT_URL $DOCKER_IMAGE $JOBNAME' | \
    tee job.yaml

kubectl apply -f job.yaml

## JOB WAITING

SLEEP_TIME=120

function getJob {
    kubectl get job -n $1 $2 -o json
}

function waitForComplete {
    # Sleep while job is active
    NAMESPACE=$1
    JOBNAME=$2
    job_active=$(getJob $NAMESPACE $JOBNAME| jq --raw-output .status.active)
    echo -e "$job_active"
    while [[ $job_active == "1" ]]
    do
        echo -e "$(date '+%Y-%m-%d %H:%M')" Job active: "$JOBNAME" ... sleeping ${SLEEP_TIME}s
        sleep $SLEEP_TIME
        job_active=$(getJob $NAMESPACE $JOBNAME| jq --raw-output .status.active)
    done

    # Check for job success
    job_succeed=$(getJob $NAMESPACE $JOBNAME | jq --raw-output .status.succeeded)
    job_fail=$(getJob $NAMESPACE $JOBNAME | jq --raw-output .status.failed)
    if [[ $job_succeed == "1" ]]
    then
        echo -e Job successful: "$JOBNAME"
    elif [[ $job_fail == "1" ]]
    then
        echo -e Job failed: "$JOBNAME"
        exit 1
    else
        echo -e Job success ambiguous: "$JOBNAME"
        exit 1
    fi
}

if [[ $detach -ne 1 ]]; then
    waitForComplete default $JOBNAME
fi


