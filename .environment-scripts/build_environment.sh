#!/bin/bash

CONDA_ENV=$1
CONDA_BASE=$(conda info --base)

case $(uname) in
    Darwin)
        packages=conda-osx-64.lock
        ;;
    Linux)
        packages=conda-linux-64.lock
        ;;
    *)
        echo "$(uname) unsupported quiting"
        exit 1
        ;;
esac

conda create -n $CONDA_ENV --file $packages 
source activate $CONDA_ENV
pip install -c constraints.txt -r pip-requirements.txt
