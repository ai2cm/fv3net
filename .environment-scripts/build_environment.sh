#!/bin/bash

set -e

CONDA_ENV=$1
CONDA_BASE=$(conda info --base)

conda env create -n $CONDA_ENV -f environment.yml
