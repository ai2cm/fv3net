#!/bin/bash

# create yaml with unique testing job name
# save job name

# submit job
kubectl apply -f integration_k8s_job.yml

# Check for successful job within 30 minutes

# else non-zero exit code