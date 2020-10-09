#!/bin/bash

echo "branch: $CIRCLE_BRANCH"
echo "tag:    $CIRCLE_TAG"

set -e
set -o pipefail

if [[ -n "$CIRCLE_SHA1" ]]
then
    CIRCLE_SHA1=$(git rev-parse HEAD)
fi

apt-get install -y make jq
make build_image_$IMAGE


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
