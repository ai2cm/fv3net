#!/bin/bash

set -e

cacheImage="$1"
shift 1

export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain


docker build \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --build-arg COMMIT_SHA_ARG="$(git rev-parse HEAD)" \
    --cache-from "$cacheImage" $@
