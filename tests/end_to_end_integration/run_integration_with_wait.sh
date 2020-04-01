#!/bin/bash

# create yaml with unique testing job name
cd tests/end_to_end_integration
rand_tag=$(openssl rand -hex 6)
job_name=$(cat submit_e2e_job_k8s.yml | yq r - metadata.name)
new_job_name=${job_name}-${rand_tag}
# save job name
yq w -i submit_e2e_job_k8s.yml metadata.name $new_job_name

# submit job
# kubectl apply -f integration_k8s_jobsubmit_e2e_job_k8s.yml

# Check for successful job within 30 minutes

# else non-zero exit code