#!/bin/bash

set -e

SLEEP_TIME=60

function waitForComplete {
    # Sleep while job is active
    job_name=$1
    NAMESPACE=$2
    timeout=$(date -ud "30 minutes" +%s)
    job_active=$(kubectl get job -n $NAMESPACE $job_name -o json | jq --raw-output .status.active)
    echo $job_active
    while [[ $(date +%s) -le $timeout ]] && [[ $job_active == "1" ]]
    do
        echo \[$(date "+%Y-%m-%d %H:%M")\] Job active: $job_name ... sleeping ${SLEEP_TIME}s
        sleep $SLEEP_TIME
        job_active=$(kubectl get job -n $NAMESPACE $job_name -o json | jq --raw-output .status.active)
    done

    # Check for job success
    job_succeed=$(kubectl get job -n $NAMESPACE $job_name -o json | jq --raw-output .status.succeeded)
    job_fail=$(kubectl get job -n $NAMESPACE $job_name -o json | jq --raw-output .status.failed)
    if [[ $job_succeed == "1" ]]
    then
        echo Job successful: $job_name
        echo Deleting job...
        kubectl delete job $job_name
    elif [[ $job_fail == "1" ]]
    then
        echo Job failed: $job_name
        exit 1
    else
        echo Job timed out or success ambiguous: $job_name
        exit 1
    fi
}

export PROGNOSTIC_RUN_IMAGE=$1
export FV3NET_IMAGE=$2

TESTDIR=tests/end_to_end_integration/
NAMESPACE=integration-tests
export JOBNAME=integration-test-$(date +%F)-$(openssl rand -hex 6)
export CONFIGMAP=integration-test-$(date +%F)-$(openssl rand -hex 6)

K8S_TEMPLATE=$TESTDIR/job.yml
E2E_TEMPLATE=$TESTDIR/end_to_end_configuration.yml

envsubst < $E2E_TEMPLATE > end-to-end.yml
envsubst < $K8S_TEMPLATE > job.yml

# use config map to make the end to end yaml available to the job
kubectl create configmap -n $NAMESPACE $CONFIGMAP --from-file=end-to-end.yml
kubectl apply -n $NAMESPACE -f job.yml
waitForComplete $JOBNAME $NAMESPACE