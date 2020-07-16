#!/bin/bash

CONDA_ENV=$1
CONDA_BASE=$(conda info --base)

source activate $CONDA_ENV || conda env create -n $CONDA_ENV -f environment.yml
