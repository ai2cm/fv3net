#!/bin/bash

CONDA_ENV=$1
CONDA_BASE=$(conda info --base)


function activateEnv {
    source $CONDA_BASE/etc/profile.d/conda.sh
    conda activate $CONDA_ENV
}


conda env create -n $CONDA_ENV -f environment.yml  2> /dev/null || \
	echo "Conda env already exists proceeding to VCM package installation"

activateEnv
conda develop . external/vcm external/fv3config

