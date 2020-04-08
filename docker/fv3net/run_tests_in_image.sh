#!/bin/bash

FV3NET_IMAGE="us.gcr.io/vcm-ml/fv3net"

set -e

if [[ -z $1 ]]; then
    VERSION_TAG="latest"
else
    VERSION_TAG=$1
fi

if [[ -z $GOOGLE_APPLICATION_CREDENTIALS ]]; then
    docker run -it $FV3NET_IMAGE:$VERSION_TAG py.test
else
    docker run -v $GOOGLE_APPLICATION_CREDENTIALS:/tmp/json.key \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/json.key \
        -it $FV3NET_IMAGE:$VERSION_TAG "py.test"
fi