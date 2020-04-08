#!/bin/bash

SLEEP_TIME=60

JOB_YML= .submit_template/submit_e2e_job_k8s.yml
# create yaml with unique testing job name
cd tests/end_to_end_integration
rand_tag=$(openssl rand -hex 6)
job_name=$(cat $JOB_YML | yq r - metadata.name)
new_job_name=${job_name}-${rand_tag}

# save with new job name and correct image tag
yq w -i $JOB_YML metadata.name $new_job_name
yq w -i $JOB_YML spec.template.spec.containers[0].image $1

# submit job
kubectl apply -f $JOB_YML

# Sleep while job is active
timeout=$(date -ud "30 minutes" +%s)
job_active=$(kubectl get job $new_job_name -o json | jq --raw-output .status.active)
while [[ $(date +%s) -le $timeout ]] && [[ $job_active == "1" ]]
do
    echo \[$(date "+%Y-%m-%d %H:%M")\] Job active: $new_job_name ... sleeping ${SLEEP_TIME}s
    sleep $SLEEP_TIME
    job_active=$(kubectl get job $new_job_name -o json | jq --raw-output .status.active)
done

# Check for job success
job_succeed=$(kubectl get job $new_job_name -o json | jq --raw-output .status.succeeded)
job_fail=$(kubectl get job $new_job_name -o json | jq --raw-output .status.failed)
if [[ $job_succeed == "1" ]]
then
    echo Job successful: $new_job_name
    echo Deleting job...
    kubectl delete job $new_job_name
elif [[ $job_fail == "1" ]]
then
    echo Job failed: $new_job_name
    exit 1
else
    echo Job timed out or success ambiguous: $new_job_name
    exit 1
fi
