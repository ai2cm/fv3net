#!/bin/bash

set -e

cacheImage="$1"
shift 1

export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain

if [[ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]]
then
    echo "Google authentication not configured. "
    echo "Please set the GOOGLE_APPLICATION_CREDENTIALS environmental variable."
    exit 1
fi

docker build \
    --secret id=gcp,src="$GOOGLE_APPLICATION_CREDENTIALS" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --build-arg COMMIT_SHA_ARG="$(git rev-parse HEAD)" 
    # --cache-from "$cacheImage" $@
