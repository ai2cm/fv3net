#!/bin/bash
set -e

CLONE_PREFIX=$1
INSTALL_PREFIX=$2
CONDA_ENV=$3
SCRIPTS=$4

CONDA_PATH=/ncrc/sw/gaea-cle7/python/3.9/anaconda-base
CONDA_SETUP="$($CONDA_PATH/bin/conda shell.bash hook 2> /dev/null)"
eval "$CONDA_SETUP"

CONDA_PREFIX=$INSTALL_PREFIX/conda
conda config --add pkgs_dirs $CONDA_PREFIX/pkgs
conda config --add envs_dirs $CONDA_PREFIX/envs
conda deactivate

conda create -n $CONDA_ENV -c conda-forge python==3.8.10 google-cloud-sdk pip pip-tools
conda activate $CONDA_ENV

echo "Compiler settings:"
echo "cc is      $(which cc)"
echo "ftn is     $(which ftn)"
echo
echo "Python settings:"
echo "conda is  $(which conda)"
echo "python is $(which python)"
echo
echo "Base enviroment packages:"
conda list
