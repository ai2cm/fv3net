#!/bin/bash
set -e

for config in $(ls end_to_end/configs/)
do
    end_to_end/generate.sh $config "$@" > manifests/end-to-end-$config.yml
done
