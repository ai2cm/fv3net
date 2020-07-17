#!/bin/bash

CONDA_ENV=$1
CONDA_BASE=$(conda info --base)

# Only create env if it does not already exist
source activate $CONDA_ENV || conda env create -n $CONDA_ENV -f environment.yml
