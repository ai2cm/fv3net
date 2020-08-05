#!/bin/bash

set -e

if [[ $# != 1 ]]
then
    echo "usage: tests/end_to_end_integration_argo/run_test.sh <version>"
    exit 1
fi

random=$(openssl rand --hex 6)
name=integration-test-$random

export VERSION=$1
export GCS_OUTPUT_URL=gs://vcm-ml-scratch/test-end-to-end-integration/$name

kubectl apply -f workflows/argo/training-rf.yaml
kubectl apply -f workflows/argo/prognostic-run.yaml
kubectl apply -f workflows/argo/nudging/nudging.yaml

cd tests/end_to_end_integration_argo

envsubst < "config_template.yaml" > "config.yaml"

argo submit --watch argo.yaml -f config.yaml --name $name
argo logs $name