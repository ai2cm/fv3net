#!/bin/bash

function usage {
  echo "Usage: "
  echo "  run_all.sh rundirs.json docker_image"
}

if [ "$#" -lt 2 ]; then
    usage
    exit 1
fi

runs=$(cat $1)
docker_image=$2
shift 2
argo submit argo.yaml -p docker-image=$docker_image -p runs="$runs" $@
