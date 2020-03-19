#!/bin/bash

CONDA_ENV=$1
CONDA_BASE=$(conda info --base)

conda env create -n $CONDA_ENV -f environment.yml  2> /dev/null || \
	echo "Conda env already exists proceeding to VCM package installation"
