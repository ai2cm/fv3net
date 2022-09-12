#!/bin/bash
set -e

REQUIREMENTS_PATH=$1
LOCAL_DEPENDENCIES=${@:2}

pip install --no-cache-dir -r $REQUIREMENTS_PATH
for dependency in $LOCAL_DEPENDENCIES
do
    pip install --no-cache-dir --no-dependencies -e $dependency
done
