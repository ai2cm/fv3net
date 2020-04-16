#!/bin/bash
set -e

for config in $(ls end_to_end/configs/)
do
    # clean up jobs
    rm -rf jobs/
    end_to_end/generate.sh $config "$@" > jobs/end-to-end-$config.yml
done
