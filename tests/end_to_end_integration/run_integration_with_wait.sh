#!/bin/bash

set -e

SLEEP_TIME=60

function waitForComplete {
    # Sleep while job is active
    jobName=$1
    NAMESPACE=$2
    timeout=$(date -ud "30 minutes" +%s)
    job_active=$(kubectl get job -n "$NAMESPACE" "$jobName" -o json | jq --raw-output .status.active)
    echo "$job_active"
    while [[ $(date +%s) -le $timeout ]] && [[ $job_active == "1" ]]
    do
        echo "$(date \"+%Y-%m-%d %H:%M\")" Job active: "$jobName" ... sleeping ${SLEEP_TIME}s
        sleep $SLEEP_TIME
        job_active=$(kubectl get job -n "$NAMESPACE" "$jobName" -o json | jq --raw-output .status.active)
    done

    # Check for job success
    job_succeed=$(kubectl get job -n "$NAMESPACE" "$jobName" -o json | jq --raw-output .status.succeeded)
    job_fail=$(kubectl get job -n "$NAMESPACE" "$jobName" -o json | jq --raw-output .status.failed)
    if [[ $job_succeed == "1" ]]
    then
        echo Job successful: "$jobName"
        echo Deleting job...
        kubectl delete job "$jobName"
    elif [[ $job_fail == "1" ]]
    then
        echo Job failed: "$jobName"
        exit 1
    else
        echo Job timed out or success ambiguous: "$jobName"
        exit 1
    fi
}

export PROGNOSTIC_RUN_IMAGE=$1
export FV3NET_IMAGE=$2

TESTDIR=$(pwd)/tests/end_to_end_integration
NAMESPACE=default
JOBNAME=integration-test-$(date +%F)-$(openssl rand -hex 6)
CONFIGMAP=integration-test-$(date +%F)-$(openssl rand -hex 6)
export JOBNAME CONFIGMAP

K8S_TEMPLATE=$TESTDIR/job_template.yml
E2E_TEMPLATE=$TESTDIR/end_to_end_template.yml


workdir=$(mktemp -d)

(
    cd "$workdir"
    envsubst < "$E2E_TEMPLATE" > end-to-end.yml
    envsubst < "$K8S_TEMPLATE" > job.yml

    # use config map to make the end to end yaml available to the job
    kubectl create configmap -n $NAMESPACE "$CONFIGMAP" --from-file=end-to-end.yml --from-file $TESTDIR/create_training_data_variable_names.yml \
        --from-file $TESTDIR/train_sklearn_model.yml
    kubectl apply -n $NAMESPACE -f job.yml
    waitForComplete "$JOBNAME" "$NAMESPACE"
)

rm -rf "$workdir"
