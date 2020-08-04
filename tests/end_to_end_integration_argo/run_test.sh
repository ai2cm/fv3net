#!/bin/bash

set -e

if [[ $# != 1 ]]
then
    echo "usage: tests/end_to_end_integration_argo/run_test.sh <version>"
    exit 1
fi

export VERSION=$1

kubectl apply -f workflows/argo/training-rf.yaml
kubectl apply -f workflows/argo/prognostic-run.yaml
kubectl apply -f workflows/argo/nudging/nudging.yaml

cd tests/end_to_end_integration_argo

envsubst < "config_template.yaml" > "config.yaml"

argo submit --wait argo.yaml -f config.yaml
