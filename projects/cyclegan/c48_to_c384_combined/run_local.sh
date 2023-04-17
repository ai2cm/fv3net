#!/bin/bash

set -e

python3 -m pdb -m fv3fit.train --validation-data-config ./validation-data.yaml ./training.yaml ./train-data.yaml ./output
