#!/bin/bash
set -e

LOCAL_DEPENDENCIES=$@

for dependency in $LOCAL_DEPENDENCIES
do
    pip install --no-cache-dir --no-dependencies -e $dependency
done
