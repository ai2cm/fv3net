#!/bin/bash

set -e

python3 -m fv3fit.train training.yaml train-data.yaml output --validation-data-config validation-data.yaml --no-wandb
