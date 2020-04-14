#!/bin/bash

SLEEP_TIME=60
JOB_YML="./tests/end_to_end_integration/submit_e2e_job_k8s.yml"
job_name=$(cat $JOB_YML | yq r - metadata.name)

# submit job
kubectl apply -f $JOB_YML

# Sleep while job is active
timeout=$(date -ud "30 minutes" +%s)
job_active=$(kubectl get job $job_name -o json | jq --raw-output .status.active)
echo $job_active
while [[ $(date +%s) -le $timeout ]] && [[ $job_active == "1" ]]
do
    echo \[$(date "+%Y-%m-%d %H:%M")\] Job active: $job_name ... sleeping ${SLEEP_TIME}s
    sleep $SLEEP_TIME
    job_active=$(kubectl get job $job_name -o json | jq --raw-output .status.active)
done

# Check for job success
job_succeed=$(kubectl get job $job_name -o json | jq --raw-output .status.succeeded)
job_fail=$(kubectl get job $job_name -o json | jq --raw-output .status.failed)
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
