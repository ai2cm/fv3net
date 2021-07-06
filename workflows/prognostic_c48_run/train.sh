#!/bin/bash
# Environmental variables used: SHA

set -e

function cleanUp {
    echo "Cleaning up $DIR"
    rm -r "$DIR"
}

DIR=$(mktemp -d)
echo "Preparing run in $DIR" > /dev/stderr
git clone https://github.com/VulcanClimateModeling/fv3net "$DIR"
trap  cleanUp EXIT
(
    cd "$DIR"
    git checkout "$SHA"
    cd workflows/prognostic_c48_run
    echo "Pulling data with dvc" > /dev/stderr
    dvc cache dir /mnt/dvc
    dvc config cache.type hardlink,symlink
    dvc pull --verbose data/training data/validation
    echo "Starting Training" > /dev/stderr
    ./train.py $@
)
