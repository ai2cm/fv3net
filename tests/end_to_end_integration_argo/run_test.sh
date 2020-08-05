#!/bin/bash

set -e

if [[ $# != 1 ]]
then
    echo "usage: tests/end_to_end_integration_argo/run_test.sh <version>"
    exit 1
fi

SLEEP_TIME=60

function getJob {
    argo get $1 -n $2 -o json
}

function waitForComplete {
    # Sleep while job is active
    jobName=$1
    NAMESPACE=$2
    timeout=$(date -ud "30 minutes" +%s)
    job_phase=$(getJob $jobName $NAMESPACE | jq -r .status.phase)
    echo "$job_phase"
    while [[ $(date +%s) -le $timeout ]] && [[ $job_phase == Running ]]
    do
        echo "$(date '+%Y-%m-%d %H:%M')" Job active: "$jobName" ... sleeping ${SLEEP_TIME}s
        sleep $SLEEP_TIME
        job_phase=$(getJob $jobName $NAMESPACE | jq -r .status.phase)
    done

    # Check job phase
    job_phase=$(getJob $jobName $NAMESPACE | jq -r .status.phase)
    if [[ $job_phase == Succeeded ]]
    then
        echo Job successful: "$jobName"
    elif [[ $job_phase == Failed ]]
    then
        echo Job failed: "$jobName"
        exit 1
    else
        echo Job timed out or success ambiguous: "$jobName"
        exit 1
    fi
}

random=$(openssl rand --hex 6)
name=integration-test-$random

export VERSION=$1
export GCS_OUTPUT_URL=gs://vcm-ml-scratch/test-end-to-end-integration/$name

kubectl apply -f workflows/argo/training-rf.yaml
kubectl apply -f workflows/argo/prognostic-run.yaml
kubectl apply -f workflows/argo/nudging/nudging.yaml

cd tests/end_to_end_integration_argo

envsubst < "config_template.yaml" > "config.yaml"

argo submit argo.yaml -f config.yaml --name $name

trap "argo logs \"$name\"" EXIT
# argo's wait/watch features are buggy, so roll our own
waitForComplete $name default
