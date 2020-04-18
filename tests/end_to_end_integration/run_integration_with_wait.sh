#!/bin/bash

set -e

if [[ $# != 1 ]]
then
    echo "usage: tests/end_to_end_integration/run_integration_with_wait.sh <version>"
    exit 1
fi

SLEEP_TIME=60

function getJob {
    kubectl get job -n $1 $2 -o json | jq '.items[0]'
}

function waitForComplete {
    # Sleep while job is active
    jobName=$1
    NAMESPACE=$2
    timeout=$(date -ud "30 minutes" +%s)
    job_active=$(getJob $NAMESPACE $jobName| jq --raw-output .status.active)
    echo "$job_active"
    while [[ $(date +%s) -le $timeout ]] && [[ $job_active == "1" ]]
    do
        echo "$(date \"+%Y-%m-%d %H:%M\")" Job active: "$jobName" ... sleeping ${SLEEP_TIME}s
        sleep $SLEEP_TIME
        job_active=$(getJob $NAMESPACE $jobName| jq --raw-output .status.active)
    done

    # Check for job success
    job_succeed=$(getJob $NAMESPACE $jobName | jq --raw-output .status.succeeded)
    job_fail=$(getJob $NAMESPACE $jobName | jq --raw-output .status.failed)
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

VERSION=$1

cd tests/end_to_end_integration/

random=$(openssl rand --hex 6)
suffix=-integration-test-$random
jobname=v1end-to-end${suffix}

cat << EOF > kustomization/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
bases:
  - "../vcm-workflow-control/base"
nameSuffix: "$suffix"
commonLabels: 
  waitForMe: "$random"
configMapGenerator:
- files:
    - time-control.yaml
  literals:
    - PROGNOSTIC_RUN_IMAGE=us.gcr.io/fv3net:$VERSION
  name: end-to-end
  behavior: merge
images:
- name: us.gcr.io/vcm-ml/fv3net
  newName: us.gcr.io/vcm-ml/fv3net
  newTag: $VERSION
EOF

echo "Running tests with this kustomization.yaml:"
cat kustomization/kustomization.yaml

kubectl apply -k kustomization
waitForComplete -lwaitForMe="$random" default
