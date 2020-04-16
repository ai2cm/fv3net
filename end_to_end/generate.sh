#!/bin/bash

set -e

if [[ $# != 1 ]]; then
    echo "Generate kubernetes manifests for an experiment"
    echo "Run this script from the root directory of the project"
    echo ""
    echo "usage:"
    echo ""
    echo "end_to_end/generate.sh <configName>"
    echo ""
    echo "configName is a folder in end_to_end/configs"
    exit 1
fi

configName="$1"

export PROGNOSTIC_RUN_IMAGE=$1
export FV3NET_IMAGE=$2

rev=$(git rev-parse HEAD)
BASE=$(pwd)/end_to_end
NAMESPACE=default
JOBNAME=$configName-$rev
CONFIGMAP=$configName-$rev

# the config directory to use inside the image
CONFIG=/etc/config
export JOBNAME CONFIGMAP CONFIG

K8S_TEMPLATE=$BASE/job_template.yml

# echo the configmap
end_to_end="$(envsubst < $BASE/end_to_end.yml)"
kubectl create configmap -n $NAMESPACE "$CONFIGMAP" \
    --dry-run \
    --from-file=$BASE/configs/$configName \
    --from-literal=PROGNOSTIC_RUN_IMAGE=$PROGNOSTIC_RUN_IMAGE \
    --from-literal=FV3NET_IMAGE=$FV3NET_IMAGE \
    --from-literal=end_to_end.yml="$end_to_end" \
    -o yaml

# echo the k8s job
echo "---"
envsubst < "$K8S_TEMPLATE"
