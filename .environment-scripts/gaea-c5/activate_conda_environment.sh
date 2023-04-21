#!/bin/bash
set -e

CONDA_ENV=$1

module load python/3.9
eval "$(conda shell.bash hook)"
echo "Activating conda environment: $CONDA_ENV"
conda activate $CONDA_ENV
