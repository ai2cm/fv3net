#!/bin/bash
set -e

CONDA_ENV=$1

module load anaconda3/2022.10
eval "$(conda shell.bash hook)"

echo "Activating conda environment: $CONDA_ENV"
conda activate $CONDA_ENV
