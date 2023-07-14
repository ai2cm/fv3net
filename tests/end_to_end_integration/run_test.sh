#!/bin/bash

set -e

SLEEP_TIME=15

function printUsage {
    echo "usage: tests/end_to_end_integration_argo/run_test.sh [-r] <registry> <version>"
    echo "-r will run the nudge-to-fine portion of the end to end test"
    exit 1
}

function getJob {
    argo get $1 -n $2
}

function terminateJob {
    argo terminate $1 -n $2
}

function getPhase {
    argo get $1 -n $2 -o json | jq -r .status.phase
}

function waitForComplete {
    # Sleep while job is active
    jobName=$1
    NAMESPACE=$2
    limitMinutes=$3
    timeout=$(date -ud "${limitMinutes} minutes" +%s)
    job_phase=$(getPhase $jobName $NAMESPACE)
    continue_phases="Running Pending null"  # job phase may be Pending or null initially
    while [[ $(date +%s) -le $timeout && "$(grep $job_phase <<< $continue_phases )" ]]
    do
        echo "$(getJob $jobName $NAMESPACE)"
        echo "$(date '+%Y-%m-%d %H:%M')" Job active: "$jobName" ... sleeping ${SLEEP_TIME}s
        sleep $SLEEP_TIME
        job_phase=$(getPhase $jobName $NAMESPACE)
    done

    # Check job phase
    job_phase=$(getPhase $jobName $NAMESPACE)
    if [[ $job_phase == Succeeded ]]
    then
        echo Job successful: "$jobName"
    elif [[ $job_phase == Failed ]]
    then
        echo Job failed: "$jobName"
        exit 1
    else
        echo "$(getJob $jobName $NAMESPACE)"
        echo Job timed out or success ambiguous: "$jobName"
        echo "$(terminateJob $jobName $NAMESPACE)"
        exit 1
    fi
}

function deployWorkflows {

    registry="$1"
    commit="$2"

    cat << EOF > kustomization.yaml
resources:
 - ../../workflows/argo
EOF

    kustomize edit set image \
        us.gcr.io/vcm-ml/prognostic_run="$registry/prognostic_run:$commit" \
        us.gcr.io/vcm-ml/fv3net="$registry/fv3net:$commit" \
        us.gcr.io/vcm-ml/post_process_run="$registry/post_process_run:$commit" \
        us.gcr.io/vcm-ml/artifacts="$registry/artifacts:$commit"

    kustomize build . | kubectl apply -f -
}

function dynamicDataConfig {

    bucket="$1"
    project="$2"
    tag="$3"

    date=$(date -u +'%F')
    # to avoid fv3net dep, we reimplement 'artifacts generate-url' here
    n2fDataPath="gs://${bucket}/${project}/${date}/${tag}-nudge-to-fine-run/fv3gfs_run"

    yq -y --arg data_path $n2fDataPath '.mapper_config.kwargs.data_path|=$data_path' \
    training-data-config.yaml > \
    training-data-config-compiled.yaml

    yq -y --arg data_path $n2fDataPath '.mapper_config.kwargs.data_path|=$data_path' \
        test-data-config.yaml > \
        test-data-config-compiled.yaml

}

registry="$1"
commit="$2"
runNudgeToFine=false
while getopts 'r' flag; do
  case "${flag}" in
    r) runNudgeToFine=true ;;
    *) printUsage
       exit 1 ;;
  esac
done
registry=${@:$OPTIND:1}
commit=${@:$OPTIND+1:1}

commitShort="${commit:0:7}"
random="$(openssl rand --hex 2)"
tag="${commitShort}-${random}"
name="integration-test-${tag}"
bucket="vcm-ml-scratch"
project="test-end-to-end-integration"

cd tests/end_to_end_integration

if [ "$runNudgeToFine" = true ] ; then
    dynamicDataConfig "${bucket}" "${project}" "${tag}"
    limitMinutes="60"
else
    cp training-data-config.yaml training-data-config-compiled.yaml
    cp test-data-config.yaml test-data-config-compiled.yaml
    limitMinutes="35"
fi

deployWorkflows "$registry" "$commit"
argo submit argo.yaml -p bucket="${bucket}" -p project="${project}" \
    -p training-data-config="$(< training-data-config-compiled.yaml)" \
    -p validation-data-config="$(< training-data-config-compiled.yaml)" \
    -p test-data-config="$(< test-data-config-compiled.yaml)" \
    -p run-nudge-to-fine="${runNudgeToFine}" \
    -p tag="${tag}" --name "$name"

trap "argo logs \"$name\" | tail -n 100" EXIT

# argo's wait/watch features are buggy, so roll our own
waitForComplete $name default $limitMinutes