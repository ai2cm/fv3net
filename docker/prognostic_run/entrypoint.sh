#!/bin/bash

set -e

# Re-compile the fortran and wrapper sources
# make -C /fv3net/external/fv3gfs-fortran/FV3
# PREFIX=/usr/local make -C /fv3net/external/fv3gfs-fortran/FV3 install
# make -C /fv3net/external/fv3gfs-wrapper build

# install the needed fv3net packages
for package in /fv3net/external/*
do
    if [[ -f $package/setup.py ]]
    then
        echo "Setting up $package"
        pip install -e "$package" --no-deps
    fi
done

exec "$@"