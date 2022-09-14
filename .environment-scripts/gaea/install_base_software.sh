#!/bin/bash
set -e

INSTALL_PREFIX=$1
CONDA_ENV=prognostic-run-2022-09-14

# We don't use module load python/3.9 since that pollutes our PATH
CONDA_PATH=/ncrc/sw/gaea-cle7/python/3.9/anaconda-base
CONDA_SETUP="$($CONDA_PATH/bin/conda shell.bash hook 2> /dev/null)"
eval "$CONDA_SETUP"

CONDA_PREFIX=$INSTALL_PREFIX/conda
conda config --add pkgs_dirs $CONDA_PREFIX/pkgs
conda config --add envs_dirs $CONDA_PREFIX/envs
conda deactivate

conda create -n $CONDA_ENV -c conda-forge python==3.8.10 pip pip-tools
conda activate $CONDA_ENV

PYTHON=`which python`
CONDA=`which conda`

echo "Compiler settings:"
echo "FC is     $FC"
echo "CC is     $CC"
echo "CXX is    $CXX"
echo "LD is     $LD"
echo "MPICC is  $MPICC"
echo
echo "Python settings:"
echo "conda is  $CONDA"
echo "python is $PYTHON"
echo
echo "Base enviroment packages:"
conda list
