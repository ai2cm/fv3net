#!/bin/bash

# create yaml with unique testing job name
cd tests/end_to_end_integration
rand_tag=$(openssl rand -hex 6)
job_name=$(cat submit_e2e_job_k8s.yml | yq r - metadata.name)
new_job_name=${job_name}-${rand_tag}
# save job name
yq w -i submit_e2e_job_k8s.yml metadata.name $new_job_name

# submit job
kubectl apply -f submit_e2e_job_k8s.yml

# Check for successful job within 30 minutes
timeout=$(date -ud "30 minutes" +%s)

while [[ $(date +%s) -le $timeout ]]; do

    job_active=$(kubectl get job $new_job_name -o json | jq --raw-output .status.active)
    job_succeed=$(kubectl get job $new_job_name -o json | jq --raw-output .status.succeded)
    if [ $job_active ]; then
        echo Job active: $new_job_name ... sleeping
        sleep 60
    elif [ ! $job_succeed ]; then
        echo Job failed: $new_job_name
        exit 1
    else
        echo Job successful: $new_job_name
    fi

done

# else non-zero exit code
echo Job timed out: $new_job_name
exit 1