#!/bin/bash
set -e

CONDA_ENV=$1

# We don't use module load python/3.9 since that pollutes our PATH
CONDA_PATH=/ncrc/sw/gaea-cle7/python/3.9/anaconda-base
CONDA_SETUP="$($CONDA_PATH/bin/conda shell.bash hook 2> /dev/null)"
eval "$CONDA_SETUP"
conda activate $CONDA_ENV
