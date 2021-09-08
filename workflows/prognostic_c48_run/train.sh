#!/bin/bash

set -e
echo "Pulling data with dvc" > /dev/stderr
dvc config cache.type hardlink,symlink
dvc pull --verbose data/training data/validation
echo "Starting Training" > /dev/stderr
python -m fv3fit.train_emulator $@
