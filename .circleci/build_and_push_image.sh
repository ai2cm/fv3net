#!/bin/bash

set -e
set -o pipefail

if [[ -n "$CIRCLE_SHA1" ]]
then
    CIRCLE_SHA1=$(git rev-parse HEAD)
fi

# Authenticate docker with GCR
echo "$ENCODED_GOOGLE_CREDENTIALS" | base64 -d > "$GOOGLE_APPLICATION_CREDENTIALS"
gcloud auth activate-service-account --key-file "$GOOGLE_APPLICATION_CREDENTIALS"
gcloud auth configure-docker

apt-get install -y make jq
git submodule update --init --recursive
make build_image_$IMAGE

echo "branch: $CIRCLE_BRANCH"
echo "tag:    $CIRCLE_TAG"

if [[ "$CIRCLE_BRANCH" == "master" ]]
then
    echo "pushing untagged images as 'latest'"
    make push_image_$IMAGE VERSION=latest
fi

if [[ -n "$CIRCLE_TAG" ]]
then
    make push_image_$IMAGE VERSION="$CIRCLE_TAG"
fi

# push all images with sha for testing later on
echo "pushing tagged images $CIRCLE_SHA1"
make push_image_$IMAGE VERSION=$CIRCLE_SHA1