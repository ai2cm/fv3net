#!/bin/bash
set -e

INSTALL_PREFIX=$1
CONDA_ENV=$2

eval "$(conda shell.bash hook)"

CONDA_PREFIX=$INSTALL_PREFIX/conda
conda config --add pkgs_dirs $CONDA_PREFIX/pkgs
conda config --add envs_dirs $CONDA_PREFIX/envs
conda deactivate

conda create -n $CONDA_ENV -c conda-forge python==3.8.10 libssh pip pip-tools
conda activate $CONDA_ENV

echo "Python settings:"
echo "conda is  $(which conda)"
echo "python is $(which python)"
echo
echo "Base enviroment packages:"
conda list
