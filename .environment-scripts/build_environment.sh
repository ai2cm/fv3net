#!/bin/bash

set -e -x

CONDA_ENV=$1
CONDA_BASE=$(conda info --base)

case $(uname) in
    Darwin)
        case $(uname -m) in
            # arm64)
            #     packages=conda-osx-arm64.lock
            #     ;;
            x86_64)
                packages=conda-osx-64.lock
                ;;
            *)
                echo "$(uname -m) on $(uname) unsupported, quitting"
                exit 1
                ;;
        esac
        ;;
    Linux)
        packages=conda-linux-64.lock
        ;;
    *)
        echo "$(uname) unsupported, quitting"
        exit 1
        ;;
esac

conda create -n $CONDA_ENV --file $packages
source activate $CONDA_ENV
pip install -c constraints.txt -r pip-requirements.txt
