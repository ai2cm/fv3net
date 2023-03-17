#!/bin/bash
set -e

CONDA_ENV=$1

CONDA_PATH=/ncrc/sw/gaea-cle7/python/3.9/anaconda-base
CONDA_SETUP="$($CONDA_PATH/bin/conda shell.bash hook 2> /dev/null)"
eval "$CONDA_SETUP"
echo "Activating conda environment: $CONDA_ENV"
conda activate $CONDA_ENV
