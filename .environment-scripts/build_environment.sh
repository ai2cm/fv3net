#!/bin/bash

CONDA_ENV=$1
CONDA_BASE=$(conda info --base)

conda env create -n $CONDA_ENV -f environment.yml && \
	conda list -n $CONDA_ENV > .circleci/.installed_build_env_deps || \
	echo "Conda env already exists proceeding to VCM package installation"
