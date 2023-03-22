#!/bin/bash
set -e

CLONE_PREFIX=$1
INSTALL_PREFIX=$2
CONDA_ENV=$3
PLATFORM_SCRIPTS=$4

bash $PLATFORM_SCRIPTS/install_bats.sh $CLONE_PREFIX/bats-core

module load anaconda3/2022.10
eval "$(conda shell.bash hook)"

CONDA_PREFIX=$INSTALL_PREFIX/conda
conda config --add pkgs_dirs $CONDA_PREFIX/pkgs
conda config --add envs_dirs $CONDA_PREFIX/envs
conda deactivate

# libssh is needed to avoid a Python "import error" upon importing
# fv3gfs.wrapper.  It installs a version of libk5crypto.so that is compatible
# with the libcrypto.so library unavoidably installed via conda,
# preventing a conflict with system versions of the two libraries on Stellar.
# See discussion in https://github.com/conda/conda/issues/10241
# for more background
conda create --yes -n $CONDA_ENV -c conda-forge python==3.8.10 libssh pip pip-tools
conda activate $CONDA_ENV

echo "Python settings:"
echo "conda is  $(which conda)"
echo "python is $(which python)"
echo
echo "Base enviroment packages:"
conda list
