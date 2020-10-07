#!/bin/bash

set -e

cacheImage="us.gcr.io/vcm-ml/$1"
shift 1

export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain

docker build \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --cache-from "$cacheImage" $@
