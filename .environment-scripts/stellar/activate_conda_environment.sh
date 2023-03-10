#!/bin/bash
set -e

CONDA_ENV=$1

eval "$(conda shell.bash hook)"

module load anaconda3/2022.10

echo "Activating conda environment: $CONDA_ENV"
conda activate $CONDA_ENV
